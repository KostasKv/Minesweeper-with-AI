# import smtplib
# import ssl

# SERVER = "smtp.office365.com"
# PORT = 587
# FROM = "yourEmail@example.com"
# TO = ["k.kvietinskas-18@student.lboro.ac.uk"] # must be a list

# SUBJECT = "Experiment progress update"
# TEXT = "This email was sent from python!"

# # Prepare actual message
# message = """From: %s\r\nTo: %s\r\nSubject: %s\r\n\

# %s
# """ % (FROM, ", ".join(TO), SUBJECT, TEXT)

# # Send the mail
# import smtplib
# server = smtplib.SMTP(SERVER)
# server.sendmail(FROM, TO, message)
# server.quit()

from tempmail import TempMail

tm = TempMail()
email = tm.get_email_address()  # v5gwnrnk7f@gnail.pw
print(tm.get_mailbox(email))  # list of emails