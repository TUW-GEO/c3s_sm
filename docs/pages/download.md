# Downloading C3S SM data via API

This packages provides a simple command line tool to download C3S Satellite
Soil Moisture data from CDS. This is just a wrapper around the [CDS python
API](https://pypi.org/project/cdsapi/) that should work for any datasets on
the CDS.

There are 2 download tools in this package. They are available after running 
``pip install c3s_sm``.
1) ``c3s_sm download`` takes a time range, version etc. and will download the
respective C3S Satellite Soil Moisture images from CDS.
2) ``c3s_sm update`` is similar, but it used to search for new images for a
locally existing record collection. It will detect the product, version and 
locally available data and download any new images that have appeared online
since the last update (e.g. you can set up a cron job to keep your records 
up-to-date)

Before any of the 2 scripts can be used, you must provide your CDS API key. 
Follow this guide: https://cds.climate.copernicus.eu/how-to-api

Make sure that 
- On Linux: You have your credentials stored in `$HOME/.cdsapirc`
- On Windows: You have your credentials stored in `%USERPROFILE%\.cdsapirc`,
%USERPROFILE% is usually located at C:\Users\Username folder
- On MacOS: Your have your credentials stored in `/Users/<USERNAME>/.cdsapirc`

Alternatively you can pass your UID and API Key (that you get from your CDS
profile page) directly with the download command (but the .cdsapirc option
is safer).

## c3s_sm download

Type ``c3s_sm download --help`` to see the full help page. A path is
always required (where the downloaded data is stored), all other arguments are 
optional.

Example command to download the daily passive product v202212 in the period from
2019-05-01 to 2019-05-10 (change the token and target path accordingly).

E.g.
```console
c3s_sm download /target/path -s 2019-05-01 -e 2019-05-10 --product passive 
--freq daily -v v202212 --cds_token xxxxxxx-xxxx-xxxx-xxxxxxxxxx
```

`--product` can be one of `active`, `combined` or `passive`. `--freq` is either
`daily`, `monthly` or `dekadal` (10-daily). For more details see the --help 
page.

This will create a subfolder for each year in the target directory and store 
downloaded images there.

Note: You don't have to provide your token if you have set up a .cdsapirc file.

```
/target/path/
├── 2019/
│   ├── C3S-SOILMOISTURE-L3S-SSMV-PASSIVE-DAILY-20190501000000-TCDR-v202212.0.0.nc
│   ├── C3S-SOILMOISTURE-L3S-SSMV-PASSIVE-DAILY-20190502000000-TCDR-v202212.0.0.nc
│   ├── ...
├── .../
```

## c3s_sm update_img

This is a simpler version of `c3s_sm download` that is applied onto an existing
data archive. The program will infer the product, sampling and version from
the existing files, detect the last available date, and download any new files
when they are available on CDS. The new files will be integrated in the local
archive.

E.g.
```console
c3s_sm update_img /target/path --cds_token xxxxxxx-xxxx-xxxx-xxxxxxxxxx
```

requires that some (previously downloaded) files are available in /target/path.
It will then check for matching new data online and download those.

Note: You don't have to provide your token if you have set up a .cdsapirc file.

