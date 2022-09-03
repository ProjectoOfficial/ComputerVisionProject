from ctypes.wintypes import SIZE
import ssl
from urllib.parse import uses_netloc
import gdown
import argparse, os
import socket
import ftputil
from ftplib import FTP 

global FILE_SIZE
global sizeWritten

sizeWritten = 0

def check_ip(ip: str):
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        return False

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

#insert host. username and password
def ftp_upload(host: str, username: str, password: str, path: str):
    ftp = FTP(host, timeout=30)
    ftp.login(user=username, passwd = password)
    print(ftp.getwelcome())
    ftp.set_pasv(False)
    print(ftp.dir())

    try: 
        print(ftp.delete(os.path.basename(path)))
    except:
        print("could not delete " + os.path.basename(path))

    file = open(path, 'rb')
    action = "STOR " + os.path.basename(path)
    print(ftp.storbinary(action, file, 1024))

    file.close()
    ftp.close()
   

#url = "https://drive.google.com/drive/folders/1rkoaZ-io_HjaeUgKoKMEQbq8vo0OQBgS"
#gdown.download_folder(url, quiet=True, use_cookies=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--download', action="store_true", default=False)
    parser.add_argument('-f', '--file', type=str)
    parser.add_argument('-U', '--username', type=str)
    parser.add_argument('-P', '--password', type=str)
    parser.add_argument('-H', '--host', type=str)
    args = parser.parse_args() 

    print(args)

    assert args.download is not None, 'specify if you want to download or upload a file'
    assert args.username is not None and args.username != "", "Invalid username"
    assert args.password is not None and args.password != "", "Invalid password"
    if not args.download:
        assert os.path.isfile(args.file), "invalid path to file (not a file?)"
        assert args.host is not None and check_ip(args.host), "Invalid host address"

    if not args.download:
        sizeWritten = 0
        totalSize = os.path.getsize(args.file)
        print('Total file size : ' + str(round(totalSize / 1024 / 1024 ,1)) + ' Mb')

        FILE_SIZE = os.path.getsize(args.file)
        ftp_upload(args.host, args.username, args.password, args.file)
    else:
        pass

    