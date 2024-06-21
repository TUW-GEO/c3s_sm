from tempfile import TemporaryDirectory
import pandas as pd
import os
import pytest
from c3s_sm.download import download_and_extract
import subprocess

def test_download_dry_run():
    with TemporaryDirectory() as outpath:
        queries = download_and_extract(
            outpath, startdate=pd.to_datetime('2022-12-25').to_pydatetime(),
            enddate=pd.to_datetime('2023-01-05').to_pydatetime(),
            version='v202212', dry_run=True,)
        assert len(queries) == 2

        assert queries[0]['cdr']['request']['type_of_record'] == 'cdr'
        assert queries[0]['cdr']['request']['year'] == ['2022']
        assert queries[0]['cdr']['request']['month'] == ['12']
        assert queries[0]['cdr']['request']['day'] == ['25', '26', '27', '28', '29', '30', '31']
        assert queries[0]['cdr']['request']['version'] == 'v202212'

        assert queries[1]['icdr']['request']['type_of_record'] == 'icdr'
        assert queries[1]['icdr']['request']['year'] == ['2023']
        assert queries[1]['icdr']['request']['month'] == ['01']
        assert queries[1]['icdr']['request']['day'] == ['01', '02', '03', '04', '05']
        assert queries[1]['icdr']['request']['version'] == 'v202212'


@pytest.mark.skipif("CDS_APIKEY" not in os.environ,
                    reason="No CDS API key found")
def test_download_with_token():
    with TemporaryDirectory() as outpath:
        args = [outpath] \
               + ['-s', '2022-06-01'] \
               + ['-e', '2022-07-31'] \
               + ['--product', 'combined'] \
               + ['--freq', 'monthly'] \
               + ['--version', 'v202212']
        subprocess.call(['c3s_sm', 'download', *args])
        files = os.listdir(os.path.join(outpath, '2022'))
        assert "C3S-SOILMOISTURE-L3S-SSMV-COMBINED-MONTHLY-20220601000000-TCDR-v202212.0.0.nc" in files
        assert "C3S-SOILMOISTURE-L3S-SSMV-COMBINED-MONTHLY-20220701000000-TCDR-v202212.0.0.nc" in files

