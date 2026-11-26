"""Download the pretrained archive linked by the official TrackNetV3 README."""
import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile

import gdown

FILE_ID = '1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA'
NAMES = ('TrackNet_best.pt',)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded official ZIP')
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent / 'ckpts'
    destination.mkdir(exist_ok=True)
    archive = args.archive or destination / 'TrackNetV3_ckpts.zip'
    if args.archive is None:
        result = gdown.download(id=FILE_ID, output=str(archive), quiet=False)
        if not result:
            raise SystemExit('Download failed. Download the official ZIP manually and use --archive PATH.')
    if not zipfile.is_zipfile(archive):
        raise SystemExit('Not a ZIP archive. Google Drive may have returned an access or quota page.')
    # Extract only the live detector, never archive-supplied paths.
    with zipfile.ZipFile(archive) as bundle:
        for name in NAMES:
            matches = [i for i in bundle.infolist() if Path(i.filename).name == name]
            if len(matches) != 1:
                raise SystemExit(f'Expected exactly one {name} in official archive')
        for name in NAMES:
            member = next(i for i in bundle.infolist() if Path(i.filename).name == name)
            output = destination / name
            with bundle.open(member) as source, output.open('wb') as target:
                shutil.copyfileobj(source, target)
            # hashlib.file_digest needs Python 3.11; the AIpro image ships an older Python.
            sha256 = hashlib.sha256()
            with output.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b''):
                    sha256.update(chunk)
            print(f'{name}: SHA256 {sha256.hexdigest()}')
    if args.archive is None:
        archive.unlink()
    print('Ready. Run run_tracking.ps1 (Windows) or run_tracking.sh (Orange Pi AIpro).')


if __name__ == '__main__':
    main()
