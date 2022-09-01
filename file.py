import ftplib
import gdown
import argparse, os

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

#insert host. username and password
session = ftplib.FTP('', '', '')

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=dir_path)
args = parser.parse_args()

os.chdir(args.path)

file = open('file.py','rb')                  # file to send
session.storbinary('STOR file.py', file)     # send the file
file.close()                                    # close file and FTP
session.quit()

#url = "https://drive.google.com/drive/folders/1rkoaZ-io_HjaeUgKoKMEQbq8vo0OQBgS"
#gdown.download_folder(url, quiet=True, use_cookies=False)
























print('File caricato')

