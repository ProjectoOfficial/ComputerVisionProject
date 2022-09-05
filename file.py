import argparse, os
import socket
from ftplib import FTP 
from tqdm import tqdm
from google_drive_downloader import GoogleDriveDownloader
from colorama import Fore
from colorama import Style

def check_ip(ip: str):
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        return False

#insert host. username and password
def ftp_upload(host: str, username: str, password: str, path: str):
    ftp = FTP(host, timeout=30)
    print(f"{Fore.BLUE}")
    print(ftp.login(user=username, passwd = password))
    print(ftp.getwelcome())
    print(f"{Fore.YELLOW}")
    ftp.set_pasv(False)
    print(ftp.dir())

    try: 
        print(ftp.delete(os.path.basename(path)))
    except:
        print("could not delete " + os.path.basename(path))

    filesize = os.path.getsize(args.file)
    file = open(path, 'rb')
    
    print(f"{Fore.MAGENTA}")
    with tqdm(unit = 'blocks', unit_scale = True, leave = False, miniters = 1, desc = 'Uploading......', total = filesize) as tqdm_instance:
        ftp.storbinary('STOR ' + os.path.basename(path), file, 2048, callback = lambda sent: tqdm_instance.update(len(sent)))

    print(f"{Style.RESET_ALL}")
    
    file.close()
    ftp.close()

def g_download(url: str, name: str, where: str):
    file_id = url.rsplit('/', 1)[-1]

    if os.path.isfile(os.path.join(where, name)):
        print(f"{Fore.YELLOW}WARNING{Style.RESET_ALL}: File already exists, download has stopped!")
        return
    
    GoogleDriveDownloader.download_file_from_google_drive(file_id=file_id,
                                                    dest_path=os.path.join(where, name),
                                                    unzip=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--download', action="store_true", default=False)
    parser.add_argument('-f', '--file', type=str)
    parser.add_argument('-U', '--username', type=str)
    parser.add_argument('-P', '--password', type=str)
    parser.add_argument('-H', '--host', type=str)
    parser.add_argument('-g', '--g_url', type=str)
    parser.add_argument('-n', '--name', type=str)
    parser.add_argument('-w', '--where', type=str)
    args = parser.parse_args()

    print(args)

    assert args.download is not None, 'specify if you want to download or upload a file'

    if not args.download:
        assert args.username is not None and args.username != "", "Invalid username"
        assert args.password is not None and args.password != "", "Invalid password"
        assert os.path.isfile(args.file), "invalid path to file (not a file?)"
        assert args.host is not None and check_ip(args.host), "Invalid host address"
        ftp_upload(args.host, args.username, args.password, args.file)
    else:
        assert args.g_url is not None and args.g_url != "", "Invalid google drive URL"
        assert args.name is not None and args.name != "", "Invalid file name"
        assert args.where is not None and os.path.isdir(args.where), "Invalid file path"
        g_download(args.g_url, args.name, args.where)

    print(f"{Fore.GREEN}program terminated successfully{Style.RESET_ALL}!")
