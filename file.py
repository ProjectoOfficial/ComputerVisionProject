from ftplib import FTP, FTP_TLS
import ipaddress
import ssl
from urllib.parse import uses_netloc
import gdown
import argparse, os
import socket

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
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS)
    session = FTP()
    session.set_pasv(True)
    session.connect(host=host)
    session.login(user=username, passwd=password)

    print(session.retrlines('LIST'))

    #file = open(path,'rb')                  # file to send
    #session.storbinary('STOR ' + path, file)     # send the file
    #file.close()                                    # close file and FTP
    session.quit()

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
        ftp_upload(args.host, args.username, args.password, args.file)
    else:
        pass