import ftplib
import gdown

#insert host. username and password
session = ftplib.FTP('', '', '')
file = open('Distance.py','rb')                  # file to send
session.storbinary('STOR Distance.py', file)     # send the file
file.close()                                    # close file and FTP
session.quit()

#url = "https://drive.google.com/drive/folders/1rkoaZ-io_HjaeUgKoKMEQbq8vo0OQBgS"
#gdown.download_folder(url, quiet=True, use_cookies=False)























print('File caricato')

