import pandas as pd
import sqlalchemy as sa

WRDS_POSTGRES_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_POSTGRES_PORT = 9737
WRDS_POSTGRES_DB = "wrds"
WRDS_CONNECT_ARGS = {"sslmode": "require", "application_name": "frds"}


class CredentialsError(PermissionError):
    pass


class NotSubscribedError(PermissionError):
    pass


class SchemaNotFoundError(FileNotFoundError):
    pass


class Connection(object):
    def __init__(self, usr, pwd, autoconnect=True):
        pguri = f"postgresql://{usr}:{pwd}@{WRDS_POSTGRES_HOST}:{WRDS_POSTGRES_PORT}/{WRDS_POSTGRES_DB}"
        self.engine = sa.create_engine(pguri, connect_args=WRDS_CONNECT_ARGS)
        if autoconnect:
            self.connect()
            self.load_library_list()

    def connect(self):
        try:
            self.connection = self.engine.connect()
        except Exception as e:
            raise e

    def close(self):
        self.connection.close()
        self.engine.dispose()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    def load_library_list(self):
        """Load the list of Postgres schemata (c.f. SAS LIBNAMEs)
        the user has permission to access."""
        self.insp = sa.inspect(self.connection)
        # print("Loading library list...")
        query = """
        WITH RECURSIVE "names"("name") AS (
            SELECT n.nspname AS "name"
                FROM pg_catalog.pg_namespace n
                WHERE n.nspname !~ '^pg_'
                    AND n.nspname <> 'information_schema')
            SELECT "name"
                FROM "names"
                WHERE pg_catalog.has_schema_privilege(
                    current_user, "name", 'USAGE') = TRUE;
        """
        cursor = self.connection.execute(query)
        self.schema_perm = [
            x[0]
            for x in cursor.fetchall()
            if not (x[0].endswith("_old") or x[0].endswith("_all"))
        ]
        # print("Loading library list... Done")

    def __check_schema_perms(self, schema):
        """
        Check the permissions of the schema.
        Raise permissions error if user does not have access.
        Raise other error if the schema does not exist.

        Else, return True

        :param schema: Postgres schema name.
        :rtype: bool

        """
        if schema in self.schema_perm:
            return True
        else:
            if schema in self.insp.get_schema_names():
                raise NotSubscribedError(
                    "You do not have permission to access"
                    "the {} library".format(schema)
                )
            else:
                raise SchemaNotFoundError("The {} library is not found.".format(schema))

    def list_libraries(self):
        """
        Return all the libraries (schemas) the user can access.

        :rtype: list

        Usage::
        >>> db.list_libraries()
        ['aha', 'audit', 'block', 'boardex', ...]
        """
        return self.schema_perm

    def list_tables(self, library):
        """
        Returns a list of all the views/tables/foreign tables within a schema.

        :param library: Postgres schema name.

        :rtype: list

        Usage::
        >>> db.list_tables('wrdssec')
        ['wciklink_gvkey', 'dforms', 'wciklink_cusip', 'wrds_forms', ...]
        """
        if self.__check_schema_perms(library):
            output = (
                self.insp.get_view_names(schema=library)
                + self.insp.get_table_names(schema=library)
                + self.insp.get_foreign_table_names(schema=library)
            )
            return output

    def __get_schema_for_view(self, schema, table):
        """
        Internal function for getting the schema based on a view
        """
        sql_code = """SELECT distinct(source_ns.nspname) AS source_schema
                      FROM pg_depend
                      JOIN pg_rewrite
                        ON pg_depend.objid = pg_rewrite.oid
                      JOIN pg_class as dependent_view
                        ON pg_rewrite.ev_class = dependent_view.oid
                      JOIN pg_class as source_table
                        ON pg_depend.refobjid = source_table.oid
                      JOIN pg_attribute
                        ON pg_depend.refobjid = pg_attribute.attrelid
                          AND pg_depend.refobjsubid = pg_attribute.attnum
                      JOIN pg_namespace dependent_ns
                        ON dependent_ns.oid = dependent_view.relnamespace
                      JOIN pg_namespace source_ns
                        ON source_ns.oid = source_table.relnamespace
                      WHERE dependent_ns.nspname = '{schema}'
                        AND dependent_view.relname = '{view}';
                    """.format(
            schema=schema, view=table
        )
        if self.__check_schema_perms(schema):
            result = self.connection.execute(sql_code)
            return result.fetchone()[0]

    def describe_table(self, library, table):
        """
        Takes the library and the table and describes all the columns
          in that table.
        Includes Column Name, Column Type, Nullable?.

        :param library: Postgres schema name.
        :param table: Postgres table name.

        :rtype: pandas.DataFrame

        Usage::
        >>> db.describe_table('wrdssec_all', 'dforms')
                    name nullable     type
              0      cik     True  VARCHAR
              1    fdate     True     DATE
              2  secdate     True     DATE
              3     form     True  VARCHAR
              4   coname     True  VARCHAR
              5    fname     True  VARCHAR
        """
        rows = self.get_row_count(library, table)
        print("Approximately {} rows in {}.{}.".format(rows, library, table))
        table_info = pd.DataFrame.from_dict(
            self.insp.get_columns(table, schema=library)
        )
        return table_info[["name", "nullable", "type"]]

    def get_row_count(self, library, table):
        """
        Uses the library and table to get the approximate
          row count for the table.

        :param library: Postgres schema name.
        :param table: Postgres table name.

        :rtype: int

        Usage::
        >>> db.get_row_count('wrdssec', 'dforms')
        16378400
        """
        if "taq" in library:
            schema = library
            print("The row count will return 0 due to the structure of TAQ")
        else:
            schema = self.__get_schema_for_view(library, table)
        if schema:
            sqlstmt = f"""
                SELECT reltuples
                  FROM pg_class r
                  JOIN pg_namespace n
                    ON (r.relnamespace = n.oid)
                  WHERE r.relkind in ('r', 'f')
                    AND n.nspname = '{schema}'
                    AND r.relname = '{table}';
                """
            try:
                result = self.connection.execute(sqlstmt)
                return int(result.fetchone()[0])
            except Exception as e:
                print(
                    "There was a problem with retrieving" "the row count: {}".format(e)
                )
                return 0
        else:
            print("There was a problem with retrieving the schema")
            return None

    def raw_sql(
        self, sql, coerce_float=True, date_cols=None, index_col=None, params=None
    ):
        """
        Queries the database using a raw SQL string.

        :param sql: SQL code in string object.
        :param coerce_float: (optional) boolean, default: True
            Attempt to convert values to non-string, non-numeric objects
            to floating point. Can result in loss of precision.
        :param date_cols: (optional) list or dict, default: None
            - List of column names to parse as date
            - Dict of ``{column_name: format string}`` where
                format string is:
                  strftime compatible in case of parsing string times or
                  is one of (D, s, ns, ms, us) in case of parsing
                    integer timestamps
            - Dict of ``{column_name: arg dict}``,
                where the arg dict corresponds to the keyword arguments of
                  :func:`pandas.to_datetime`
        :param index_col: (optional) string or list of strings,
          default: None
            Column(s) to set as index(MultiIndex)
        :param params: parameters to SQL query, if parameterized.

        :rtype: pandas.DataFrame

        Usage ::
        # Basic Usage
        >>> data = db.raw_sql('select cik, fdate, coname from wrdssec_all.dforms;', date_cols=['fdate'], index_col='cik')
        >>> data.head()
            cik        fdate       coname
            0000000003 1995-02-15  DEFINED ASSET FUNDS MUNICIPAL INVT TR FD NEW Y...
            0000000003 1996-02-14  DEFINED ASSET FUNDS MUNICIPAL INVT TR FD NEW Y...
            0000000003 1997-02-19  DEFINED ASSET FUNDS MUNICIPAL INVT TR FD NEW Y...
            0000000003 1998-03-02  DEFINED ASSET FUNDS MUNICIPAL INVT TR FD NEW Y...
            0000000003 1998-03-10  DEFINED ASSET FUNDS MUNICIPAL INVT TR FD NEW Y..
            ...

        # Parameterized SQL query
        >>> parm = {'syms': ('A', 'AA', 'AAPL'), 'num_shares': 50000}
        >>> data = db.raw_sql('select * from taqmsec.ctm_20030910 where sym_root in %(syms)s and size > %(num_shares)s', params=parm)
        >>> data.head()
                  date           time_m ex sym_root sym_suffix tr_scond      size   price tr_stopind tr_corr     tr_seqnum tr_source tr_rf
            2003-09-10  11:02:09.485000  T        A       None     None  211400.0  25.350          N      00  1.929952e+15         C  None
            2003-09-10  11:04:29.508000  N        A       None     None   55500.0  25.180          N      00  1.929952e+15         C  None
            2003-09-10  15:08:21.155000  N        A       None     None   50500.0  24.470          N      00  1.929967e+15         C  None
            2003-09-10  16:10:35.522000  T        A       None        B   71900.0  24.918          N      00  1.929970e+15         C  None
            2003-09-10  09:35:20.709000  N       AA       None     None  108100.0  28.200          N      00  1.929947e+15         C  None
        """
        try:
            return pd.read_sql_query(
                sql,
                self.connection,
                coerce_float=coerce_float,
                parse_dates=date_cols,
                index_col=index_col,
                params=params,
            )
        except sa.exc.ProgrammingError as e:
            raise e

    def get_table(
        self,
        library,
        table,
        obs=-1,
        offset=0,
        columns=None,
        coerce_float=None,
        index_col=None,
        date_cols=None,
    ):
        """
        Creates a data frame from an entire table in the database.

        :param sql: SQL code in string object.
        :param library: Postgres schema name.

        :param obs: (optional) int, default: -1
            Specifies the number of observations to pull from the table.
            An integer less than 0 will return the entire table.
        :param offset: (optional) int, default: 0
            Specifies the starting point for the query.
            An offset of 0 will start selecting from the beginning.
        :param columns: (optional) list or tuple, default: None
            Specifies the columns to be included in the output data frame.
        :param coerce_float: (optional) boolean, default: True
            Attempt to convert values to non-string, non-numeric objects
            to floating point. Can result in loss of precision.
        :param date_cols: (optional) list or dict, default: None
            - List of column names to parse as date
            - Dict of ``{column_name: format string}``
                where format string is
                  strftime compatible in case of parsing string times or
                  is one of (D, s, ns, ms, us) in case of parsing
                    integer timestamps
            - Dict of ``{column_name: arg dict}``,
                where the arg dict corresponds to the keyword arguments of
                  :func:`pandas.to_datetime`
        :param index_col: (optional) string or list of strings,
          default: None
            Column(s) to set as index(MultiIndex)

        :rtype: pandas.DataFrame

        Usage ::
        >>> data = db.get_table('wrdssec_all', 'dforms', obs=1000, columns=['cik', 'fdate', 'coname'])
        >>> data.head()
            cik        fdate       coname
            0000000003 1995-02-15  DEFINED ASSET FUNDS MUNICIPAL INVT TR FD NEW Y...
            0000000003 1996-02-14  DEFINED ASSET FUNDS MUNICIPAL INVT TR FD NEW Y...
            0000000003 1997-02-19  DEFINED ASSET FUNDS MUNICIPAL INVT TR FD NEW Y...
            0000000003 1998-03-02  DEFINED ASSET FUNDS MUNICIPAL INVT TR FD NEW Y...
            0000000003 1998-03-10  DEFINED ASSET FUNDS MUNICIPAL INVT TR FD NEW Y..
            ...

        """
        if obs < 0:
            obsstmt = ""
        else:
            obsstmt = " LIMIT {}".format(obs)
        if columns is None:
            cols = "*"
        else:
            cols = ",".join(columns)
        if self.__check_schema_perms(library):
            sqlstmt = (
                "SELECT {cols} FROM {schema}.{table} {obsstmt} OFFSET {offset};".format(
                    cols=cols,
                    schema=library,
                    table=table,
                    obsstmt=obsstmt,
                    offset=offset,
                )
            )
            return self.raw_sql(
                sqlstmt,
                coerce_float=coerce_float,
                index_col=index_col,
                date_cols=date_cols,
            )
