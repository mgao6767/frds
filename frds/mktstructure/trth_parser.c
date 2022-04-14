#include <Python.h>
#if PY_MAJOR_VERSION >= 3
#define PY3K
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#ifdef _WIN32
#include <direct.h>
#endif
#define F_OK 0 /* Test for existence.  */
#define DEBUG 0
#define DEBUG_DETAIL 0
#define MKDIR_MODE 0775
#define BUFSIZE 1024
#define FIELD_DATETIME "Date-Time"
#define FIELD_RIC "#RIC"
#define FIELD_GMTOFFSET "GMT Offset"
#define PATH_MAX_STRING_SIZE 512
// The expected number of transactions for a security in a day
// This needs to be dynamically managed.
#define CHUNK_LENGTH 10000000
#define FOR_EACH_TOKEN(CTX, I, S, D)                         \
  for (CTX = (S), (I) = get_next_token(&(CTX), D); (I) != 0; \
       (I) = get_next_token(&(CTX), D))

char *get_local_date(char *datetimeISO, int GMT_offset);
char *get_next_token(char **context, const char *delim);
struct Fields *get_meta_info(FILE *file, int close_after_read);
void process(FILE *file, struct Fields *meta, const char *output_dir,
             int replace);
int save_chunk(char *RIC, char *local_date, const char *fields, char **chunk,
               size_t chunk_size, const char *output_dir, int replace);
int mkdir_p(const char *dir, const int mode);

// Store meta info of the file.
struct Fields {
  // Total number of fields.
  int lengthOfFields;
  // Location of Date-Time in the fields.
  int locFieldDateTime;
  // Location of RIC in the fields.
  int locFieldRIC;
  // Location of GMT Offset in the fields.
  int locFieldGMTOffset;
  // All fields as appeared in the origional file.
  const char *fields;
};

// Extract meta info from the file.
struct Fields *get_meta_info(FILE *file, int close_after_read) {
  if (file == NULL) exit(EXIT_FAILURE);
  // Init a buffer
  static char buffer[BUFSIZE];
  // Read first line from the file.
  fgets(buffer, BUFSIZE, file);
  struct Fields *meta = malloc(sizeof(struct Fields));
  meta->lengthOfFields = 0;
  meta->fields = strdup(buffer);
  char *context, *field;
  FOR_EACH_TOKEN(context, field, buffer, ",") {
    if (strcmp(field, FIELD_DATETIME) == 0)
      meta->locFieldDateTime = meta->lengthOfFields;
    if (strcmp(field, FIELD_RIC) == 0) meta->locFieldRIC = meta->lengthOfFields;
    if (strcmp(field, FIELD_GMTOFFSET) == 0)
      meta->locFieldGMTOffset = meta->lengthOfFields;
    meta->lengthOfFields++;
  }
  if (close_after_read) fclose(file);
#if DEBUG_DETAIL
  printf("meta->locFieldRIC %d\n", meta->locFieldRIC);
  printf("meta->locFieldDateTime %d\n", meta->locFieldDateTime);
  printf("meta->locFieldGMTOffset %d\n", meta->locFieldGMTOffset);
  printf("meta->lengthOfFields %d\n", meta->lengthOfFields);
  printf("meta->fields %s\n", meta->fields);
#endif
  return meta;
}

int main(int argc, char const *argv[]) {
  const char *input = NULL;
  const char *output_dir = NULL;
  int replace = 0;
  if (argc == 4) {
    // Input file name.
    input = argv[1];
    // Output directory.
    output_dir = argv[2];
    // Wether to replace existing output files.
    replace = atoi(argv[3]);
    puts(input);
    puts(output_dir);
  } else
    return EXIT_FAILURE;

  FILE *file = fopen(input, "r");
  if (file == NULL) exit(EXIT_FAILURE);

  // Get meta info.
  struct Fields *meta = get_meta_info(file, 0);

  // Start processing line by line.
  process(file, meta, output_dir, replace);

  // Closing up.
  free(meta);
  meta = NULL;
  fclose(file);
#if DEBUG
  puts("Finished parsing!\n");
#endif
  return EXIT_SUCCESS;
}

// Process the data part of the file.
void process(FILE *file, struct Fields *meta, const char *output_dir,
             int replace) {
  if (file == NULL) exit(EXIT_FAILURE);
  // Init a buffer, context and field pointers.
  static char buffer[BUFSIZE];
  // Keep track of last RIC and local date.
  char *lastRIC = NULL, *lastLocalDate = NULL;
  // Keep track of current RIC and local date.
  char *thisRIC = NULL, *thisDateTime = NULL;
  char *thisGMTOffset = NULL, *thisLocalDate = NULL;
  // Pointers to use in parsing each line.
  char *context = NULL, *field = NULL;
  // Array of pointers to transactions of a security in a day (a chunk).
  // This needs to be dynamically managed, but for now let's make it static.
  char **chunk = malloc(sizeof(char *) * CHUNK_LENGTH);
  // Total number of transactions (lines) of a chunk.
  size_t numTransactions = 0;
  // Read line by line.
  while (fgets(buffer, BUFSIZE, file)) {
    // Get a pointer to current line.
    char *current_data = malloc(sizeof(char) * strlen(buffer) + 1);
    strcpy(current_data, buffer);
    // Current location of the field.
    int loc = 0;
    // All identifiers found.
    int identifiers_found = 0;
    // Find out the RIC, datetime and GMT offset of current line.
    FOR_EACH_TOKEN(context, field, buffer, ",") {
      if (loc == meta->locFieldRIC) {
        thisRIC = strdup(field);
        identifiers_found++;
      } else if (loc == meta->locFieldDateTime) {
        thisDateTime = field;
        identifiers_found++;
      } else if (loc == meta->locFieldGMTOffset) {
        thisGMTOffset = field;
        identifiers_found++;
      }
      loc++;
      // Break parsing the row when all identifiers are found.
      if (identifiers_found == 3) break;
    }
    // Compute local date.
    thisLocalDate = get_local_date(thisDateTime, atoi(thisGMTOffset));
    // Extend chunk if necessary.
    if (numTransactions > CHUNK_LENGTH)
      chunk = realloc(chunk, sizeof(char *) * numTransactions * 1.2);
    // We have current RIC and local date now.
    // This is the first data row.
    if (lastRIC == NULL) {
      lastRIC = thisRIC;
      lastLocalDate = thisLocalDate;
      // Add this data row to chunk.
      chunk[numTransactions++] = current_data;
    } else
    // Rest of the data rows.
    {
      // If current RIC is same as last RIC.
      if (strcmp(thisRIC, lastRIC) == 0) {
        // If current local date is same as last local date and same RIC.
        if (strcmp(thisLocalDate, lastLocalDate) == 0) {
          // Same RIC and local date as previous rows.
          // Add this data row to chunk.
          chunk[numTransactions++] = current_data;
          free(thisLocalDate);
          // Need to free thisRIC as well. Since next iteration will make a new
          // thisRIC.
          free(thisRIC);
        } else
        // Current local date is different from last local date but same RIC.
        {
          // Valid row.
          // When there is a different date, old chunk is saved.
          save_chunk(lastRIC, lastLocalDate, meta->fields, chunk,
                     numTransactions, output_dir, replace);
          numTransactions = 0;
          // Add this data row to chunk.
          chunk[numTransactions++] = current_data;
          free(lastLocalDate);
          lastLocalDate = thisLocalDate;
        }
      } else
      // Current RIC is different from last RIC.
      {
        // When there is a different SIC, old chunk is saved.
        save_chunk(lastRIC, lastLocalDate, meta->fields, chunk, numTransactions,
                   output_dir, replace);
        numTransactions = 0;
        // Add this data row to chunk.
        chunk[numTransactions++] = current_data;
        // New RIC also means new local date.
        free(lastRIC);
        lastRIC = thisRIC;
        free(lastLocalDate);
        lastLocalDate = thisLocalDate;
      }
    }
  }
  // Save the remaining part after finishing reading the entire file.
  // Case 1: the file contains only one RIC and one local date.
  // Case 2: the last valid chunk.
  if (numTransactions > 0)
    save_chunk(thisRIC, thisLocalDate, meta->fields, chunk, numTransactions,
               output_dir, replace);

  // Free chunk.
  free(chunk);
  chunk = NULL;
}

// Compute the local date based on GMTUTC and GMT offset.
char *get_local_date(char *datetimeISO, int GMT_offset) {
  int Y, M, d, h, m;
  float s;
  sscanf(datetimeISO, "%d-%d-%dT%d:%d:%f+00:00", &Y, &M, &d, &h, &m, &s);
  // Compute local date. DOUBLE CHECK！
  struct tm a = {.tm_year = Y - 1900,
                 .tm_mon = M - 1,
                 .tm_mday = d,
                 .tm_sec = s,
                 .tm_min = m,
                 .tm_hour = h + GMT_offset - 1};
  // const time_t local_time = mktime(&a);
  const int len = strlen("1994-10-21");
  char *date = malloc(len + 1);
  strftime(date, len + 1, "%Y-%m-%d", &a);
  return date;
}

int save_chunk(char *RIC, char *local_date, const char *fields, char **chunk,
               size_t chunk_size, const char *output_dir, int replace) {
  char path[PATH_MAX_STRING_SIZE];
  strcpy(path, output_dir);
  const int len = strnlen(path, PATH_MAX_STRING_SIZE);
  if (path[len - 1] == '/' || path[len - 1] == '\\')
    strcat(path, RIC);
  else {
#ifdef _WIN32
    path[len] = '\\';
#else
    path[len] = '/';
#endif
    path[len + 1] = '\0';
    strcat(path, RIC);
  }
  const int path_len = strlen(path) + strlen(local_date) + 9;
  char *output_file = malloc(sizeof(char) * path_len);
#ifdef _WIN32
  snprintf(output_file, path_len, "%s\\%s.csv", path, local_date);
#else
  snprintf(output_file, path_len, "%s/%s.csv", path, local_date);
#endif
#if DEBUG
  printf("Path: %s, output_file: %s\n", path, output_file);
#endif
  if (mkdir_p(path, MKDIR_MODE) == -1) {
    puts("Cannot mkdir!");
    return -1;
  };
  char *mode = NULL;
  if (replace == 1)
    mode = "wb";
  else
    mode = "ab";
  FILE *outputFile;
  if (access(output_file, F_OK) != -1) {
    // File exists.
    outputFile = fopen(output_file, mode);
    // Write header row if file exists but we're to overwrite it.
    if (replace == 1) fprintf(outputFile, "%s", fields);
  } else {
    // File doesn't exist. Write header row.
    outputFile = fopen(output_file, mode);
    fprintf(outputFile, "%s", fields);
  }
  for (size_t i = 0; i < chunk_size; i++) {
    fprintf(outputFile, "%s", chunk[i]);
    free(chunk[i]);
    chunk[i] = NULL;
  }
  fclose(outputFile);
  printf("Saved chunk RIC: %s, local date: %s at %s (%ld rows)\n", RIC,
         local_date, output_file, chunk_size);
  free(output_file);
  return 0;
}

char *get_next_token(char **context, const char *delim) {
  char *ret;

  /* A null context indicates no more tokens. */
  if (*context == 0) return 0;

  /* Skip delimiters to find start of token */
  ret = (*context += strspn(*context, delim));

  /* skip to end of token */
  *context += strcspn(*context, delim);

  /* If the token has zero length, we just
   skipped past a run of trailing delimiters, or
   were at the end of the string already.
   There are no more tokens. */

  if (ret == *context) {
    *context = 0;
    return 0;
  }

  /* If the character past the end of the token is the end of the string,
   set context to 0 so next time we will report no more tokens.
   Otherwise put a 0 there, and advance one character past. */

  if (**context == 0) {
    *context = 0;
  } else {
    **context = 0;
    (*context)++;
  }

  return ret;
}

int mkdir_p(const char *dir, const int mode) {
  char tmp[PATH_MAX_STRING_SIZE];
  char *p = NULL;
  struct stat sb;
  size_t len;

  /* copy path */
  len = strnlen(dir, PATH_MAX_STRING_SIZE);
  if (len == 0 || len == PATH_MAX_STRING_SIZE) {
    return -1;
  }
  memcpy(tmp, dir, len);
  tmp[len] = '\0';

  /* remove trailing slash */
#ifdef _WIN32
  if (tmp[len - 1] == '\\')
#else
  if (tmp[len - 1] == '/')
#endif
  {
    tmp[len - 1] = '\0';
  }

  /* check if path exists and is a directory */
  if (stat(tmp, &sb) == 0) {
    if (S_ISDIR(sb.st_mode)) {
      return 0;
    }
  }

  /* recursive mkdir */
  for (p = tmp + 1; *p; p++) {
#ifdef _WIN32
    if (*p == '\\')
#else
    if (*p == '/')
#endif
    {
      *p = 0;
      /* test path */
      if (stat(tmp, &sb) != 0) {
        /* path does not exist - create directory */
#ifdef _WIN32
        if (_mkdir(tmp) < 0)
#else
        if (mkdir(tmp, mode) < 0)
#endif
        {
          return -1;
        }
      } else if (!S_ISDIR(sb.st_mode)) {
        /* not a directory */
        return -1;
      }
#ifdef _WIN32
      *p = '\\';
#else
      *p = '/';
#endif
    }
  }
  /* test path */
  if (stat(tmp, &sb) != 0) {
    /* path does not exist - create directory */
#ifdef _WIN32
    if (_mkdir(tmp) < 0)
#else
    if (mkdir(tmp, mode) < 0)
#endif
    {
      return -1;
    }
  } else if (!S_ISDIR(sb.st_mode)) {
    /* not a directory */
    return -1;
  }
  return 0;
}

static PyObject *trth_parser_wrapper(PyObject *self, PyObject *args) {
  char *input_file, *output_dir, *replace = NULL;
  /* Parse arguments */
  if (!PyArg_ParseTuple(args, "sss", &input_file, &output_dir, &replace)) {
    return NULL;
  }

  char const *args_to_main[4] = {"trth_parser", input_file, output_dir,
                                 replace};
  return PyLong_FromLong(main(4, args_to_main));
};

static PyMethodDef trth_parser_methods[] = {
    {"parse_to_data_dir", trth_parser_wrapper, METH_VARARGS,
     "C parser for downloaded intraday tick data from TRTH."},
    {NULL, NULL, 0, NULL}};

#ifdef PY3K
// module definition structure for python3
static struct PyModuleDef trth_parser = {
    PyModuleDef_HEAD_INIT, "trth_parser",
    "Python interface for the C parser for TRTH data", -1, trth_parser_methods};
// module initializer for python3
PyMODINIT_FUNC PyInit_trth_parser() { return PyModule_Create(&trth_parser); }
#else
// module initializer for python2
PyMODINIT_FUNC init_trth_parser() {
  Py_InitModule3("trth_parser", trth_parser_methods,
                 "Python interface for the C parser for TRTH data");
}
#endif
