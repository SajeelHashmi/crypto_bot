import smtplib
from email.message import EmailMessage
import os
from dotenv import load_dotenv
load_dotenv()

# whenever a person signs in send them a message
sender = os.environ.get('EMAIL_SENDER')
password = os.environ.get("EMAIL_PASS")



import requests
headers = {'Content-type': 'application/json'}

base_url = 'https://338c7f40-cd15-477d-9403-67afc96d8367-00-2hbz1c498d3bb.sisko.replit.dev'


def sendEmail(msg,subject = "Trade Signal"):

    em = EmailMessage()
    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)
    # start TLS for security
    s.starttls()
    try:
        # Authentication
        s.login(sender, password)
    except:
        print('Incorrect Username or Password:\n')
        return 
    em["To"] = 'muhammad.asad85@ce.ceme.edu.pk'
    message = msg
    em["From"] = sender
    em["Subject"] = subject
    em.set_content(message)
    s.send_message(em)
    s.quit()

    print("email sent")
    return


def sendEmailTest(msg,tradeId=None):
    try:
        
        em = EmailMessage()
        # creates SMTP session
        s = smtplib.SMTP('smtp.gmail.com', 587)
        # start TLS for security
        s.starttls()
        try:
            # Authentication
            s.login(sender, password)
        except:
            print('Incorrect Username or Password:\n')
            return 
        em["To"] = 'kajkgakljfdklafjkalj@jklafjakjfjkls.com'
        message = msg
        em["From"] = sender
        em.set_content(message)
        s.send_message(em)
        s.quit()



        print("email sent")
    except:
        return
    
if __name__ =="__main__":
    sendEmailTest("TEST")  