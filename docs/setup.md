# Setup

After installation, a folder `frds` will be created under your user's home directory, which contains a `data` folder, a `result` folder and a default configuration file `config.ini`:

```ini
[Paths]
base_dir: ~/frds
data_dir: ${base_dir}/data
result_dir: ${base_dir}/result

[Login]
wrds_username: 
wrds_password: 
```

You need to enter your WRDS username and password under the login section if you wish to use [`frds`](/) via CLI. Alternatively, you can leave them empty and enter manually when using the GUI.
