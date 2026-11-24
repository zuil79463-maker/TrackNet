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
            with output.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            print(f'{name}: SHA256 {digest}')
    if args.archive is None:
        archive.unlink()
    print('Ready. Run run_tracking.ps1 to open the camera.')


if __name__ == '__main__':
    main()
