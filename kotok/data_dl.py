import os
import zipfile
import logging
import requests

entries = {
    'airbnb_review.txt': 'https://github.com/yiunsr/boddari/raw/refs/heads/main/boddari/txt/airbnb_review.txt.zip',
    'book_review01.txt': 'https://github.com/yiunsr/boddari/raw/refs/heads/main/boddari/txt/book_review01.txt',
    'encyclopedia01.01.txt': 'https://github.com/yiunsr/boddari/raw/refs/heads/main/boddari/txt/encyclopedia01.01.txt.zip',
    'fairytale01.txt': 'https://github.com/yiunsr/boddari/raw/refs/heads/main/boddari/txt/fairytale01.txt',
    'policy_news.01.txt': 'https://github.com/yiunsr/boddari/raw/refs/heads/main/boddari/txt/policy_news.01.txt.zip',
    'nsmc.txt': 'https://github.com/e9t/nsmc/raw/refs/heads/master/ratings.txt',
    'unlabeled_comments_1.txt': 'https://github.com/kocohub/korean-hate-speech/raw/refs/heads/master/unlabeled/unlabeled_comments_1.txt',
}

def data_dl(out_dir, **_kwargs):
    """
    Downloads all the data files to the specified directory. Unzips if necessary.
    """
    os.makedirs(out_dir, exist_ok=True)

    for filename, url in entries.items():
        logging.info(f'Downloading {filename} from {url}')

        out_path = os.path.join(out_dir, filename)

        if url.endswith('.zip'):
            out_path = out_path + '.zip'

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(out_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # Unzip the file if it has a .zip extension
        if out_path.endswith('.zip'):        
            logging.info(f'Unzipping {filename}')
            with zipfile.ZipFile(out_path, 'r') as z:
                z.extractall(out_dir)

            os.remove(out_path)

        # convert nsmc to plain text
        if filename == 'nsmc.txt':
            with open(os.path.join(out_dir, 'nsmc.txt'), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            with open(os.path.join(out_dir, 'nsmc.txt'), 'w', encoding='utf-8') as f:
                for i, line in enumerate(lines):
                    if i == 0:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) != 3:
                        continue
                    f.write(parts[1])
                    f.write('\n')

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    data_dl(sys.argv[1] if len(sys.argv) > 1 else '.')
