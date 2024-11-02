import logging
import psycopg2
from datetime import datetime
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from azure.functions import func
import os

# Constants
DB_CONNECTION_STRING = "dbname=techconfdb user=nvnhan@project3-udacity password=Nhan12344321 host=project3-udacity.postgres.database.azure.com"

def main(msg: func.ServiceBusMessage):
    try:
        # Get the notification ID from the message
        notification_id = int(msg.get_body().decode('utf-8'))
        logging.info('Message: %s', notification_id)

        # Connect to the database
        with psycopg2.connect(DB_CONNECTION_STRING) as connection:
            with connection.cursor() as cursor:
                # Get notification message and subject from the database using the notification_id
                cursor.execute("SELECT message, subject FROM notification WHERE id = %s;", (notification_id,))
                notification = cursor.fetchone()

                if notification:
                    message, subject = notification

                    # Get attendees' email and name
                    cursor.execute("SELECT first_name, last_name, email FROM attendee;")
                    attendees = cursor.fetchall()

                    # Loop through each attendee and send a personalized email
                    for attendee in attendees:
                        attendee_first_name, attendee_last_name, attendee_email = attendee
                        personalized_subject = f'Hello {attendee_first_name} - {subject}'
                        mail = Mail(
                            from_email='infopr3@techconf.com',
                            to_emails=attendee_email,
                            subject=personalized_subject,
                            plain_text_content=message
                        )
                        send_email(mail)

                    # Update the notification table
                    notification_completed_date = datetime.utcnow()
                    notification_status = f'Notified {len(attendees)} attendees'
                    cursor.execute("UPDATE notification SET status = %s, completed_date = %s WHERE id = %s;",
                                   (notification_status, notification_completed_date, notification_id))
                    connection.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(error)

def send_email(mail):
    sg = SendGridAPIClient(os.environ['SENDGRID_API_KEY'])
    sg.send(mail)
