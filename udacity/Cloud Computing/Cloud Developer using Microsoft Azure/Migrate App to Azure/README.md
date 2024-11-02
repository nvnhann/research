# TechConf Registration Website

## Project Overview
The TechConf website allows attendees to register for an upcoming conference. Administrators can also view the list of attendees and notify all attendees via a personalized email message.

The application is currently working but the following pain points have triggered the need for migration to Azure:
 - The web application is not scalable to handle user load at peak
 - When the admin sends out notifications, it's currently taking a long time because it's looping through all attendees, resulting in some HTTP timeout exceptions
 - The current architecture is not cost-effective 

In this project, you are tasked to do the following:
- Migrate and deploy the pre-existing web app to an Azure App Service
- Migrate a PostgreSQL database backup to an Azure Postgres database instance
- Refactor the notification logic to an Azure Function via a service bus queue message

## Dependencies

You will need to install the following locally:
- [Postgres](https://www.postgresql.org/download/)
- [Visual Studio Code](https://code.visualstudio.com/download)
- [Azure Function tools V3](https://docs.microsoft.com/en-us/azure/azure-functions/functions-run-local?tabs=windows%2Ccsharp%2Cbash#install-the-azure-functions-core-tools)
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)
- [Azure Tools for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode.vscode-node-azure-pack)

## Project Instructions

### Part 1: Create Azure Resources and Deploy Web App
1. Create a Resource group
2. Create an Azure Postgres Database single server
   - Add a new database `techconfdb`
   - Allow all IPs to connect to database server
   - Restore the database with the backup located in the data folder
3. Create a Service Bus resource with a `notificationqueue` that will be used to communicate between the web and the function
   - Open the web folder and update the following in the `config.py` file
      - `POSTGRES_URL`
      - `POSTGRES_USER`
      - `POSTGRES_PW`
      - `POSTGRES_DB`
      - `SERVICE_BUS_CONNECTION_STRING`
4. Create App Service plan
5. Create a storage account
6. Deploy the web app

### Part 2: Create and Publish Azure Function
1. Create an Azure Function in the `function` folder that is triggered by the service bus queue created in Part 1.

      **Note**: Skeleton code has been provided in the **README** file located in the `function` folder. You will need to copy/paste this code into the `__init.py__` file in the `function` folder.
      - The Azure Function should do the following:
         - Process the message which is the `notification_id`
         - Query the database using `psycopg2` library for the given notification to retrieve the subject and message
         - Query the database to retrieve a list of attendees (**email** and **first name**)
         - Loop through each attendee and send a personalized subject message
         - After the notification, update the notification status with the total number of attendees notified
2. Publish the Azure Function

### Part 3: Refactor `routes.py`
1. Refactor the post logic in `web/app/routes.py -> notification()` using servicebus `queue_client`:
   - The notification method on POST should save the notification object and queue the notification id for the function to pick it up
2. Re-deploy the web app to publish changes

## Monthly Cost Analysis
Complete a month cost analysis of each Azure resource to give an estimate total cost using the table below:

| Azure Resource              | Service Tier            | Monthly Cost Estimation                    |
| ---------------------------- | ----------------------- | ------------------------------------------ |
| **Azure Postgres Database**  | Premium tier            | Estimate depends on DTU, storage, and backup strategy. |
| **Azure Service Bus**       | Premium tier            | Estimate based on message units, topics, and queues. |
| **Azure Functions (Premium Plan)** | Premium Plan     | Pay-per-execution, resource usage, and dedicated resources. |
| **SendGrid Email Service**  | Premium Plan            | Estimate depends on email volume and features. |
| **Storage Account**         | Standard (Hot tier)     | Estimate depends on data storage and transactions.

Here are some considerations for a production-level estimation:

1. **Azure Postgres Database**:
   - Choose a Premium tier to ensure better performance and reliability.
   - Consider the number of DTUs (Database Throughput Units), storage, and backup/replication strategy for high availability.

2. **Azure Service Bus**:
   - Use Premium tier to ensure higher scalability, performance, and advanced features.
   - Estimate based on the number of message units, topics, and queues.

3. **Azure Functions (Premium Plan)**:
   - Opt for a Premium Plan to ensure dedicated resources for your functions and better performance.
   - Estimate the number of function executions, execution time, and resource usage.

4. **SendGrid Email Service**:
   - Consider a Premium Plan to access advanced features and higher sending limits.
   - Estimate the number of emails you plan to send per month.

5. **Storage Account**:
   - Choose a Standard tier with a Hot access tier for better performance.
   - Estimate the amount of data storage and transactions required.
   
## Architecture Selection

### Azure Web App

For the Azure Web App component of our project, we have chosen a multi-tier architecture that includes the following elements:

1. **Frontend**: We are using the Azure Web App as the hosting platform for our application's frontend. This allows us to benefit from Azure's managed web hosting service, which provides scalability, load balancing, and a secure environment for our web application.

2. **Backend API**: To separate concerns and maintain a scalable architecture, we have chosen to implement our backend API as a separate component. This API is responsible for serving data to the frontend, handling user authentication, and managing the core business logic.

3. **Database**: We utilize Azure Postgres Database for data storage. This is a fully managed, relational database service, which offers high availability and reliability. The choice of the Premium tier ensures that we have sufficient resources for production-level workloads.

The reasons for this architecture selection are as follows:

- **Scalability**: By separating the frontend and backend, we can scale them independently. The Azure Web App's auto-scaling capabilities ensure that we can handle increased web traffic efficiently.

- **Security**: Azure Web App provides built-in security features, and by using Azure Active Directory for authentication, we can ensure a high level of security for our application.

- **Managed Services**: Utilizing Azure managed services, like Azure Postgres Database, reduces operational overhead and ensures that we have access to the latest features, updates, and security patches.

### Azure Function

For the Azure Function component of our project, we have chosen a serverless architecture to handle asynchronous tasks and background processing. Azure Functions are employed to:

1. **Process Service Bus Messages**: We use Azure Functions with a Service Bus trigger to process messages asynchronously. This is especially useful for background tasks like sending notifications.

2. **Email Notification**: Azure Functions enable cost-effective and scalable email notification processing. The Premium Plan offers dedicated resources to ensure efficient execution.

The reasons for this architecture selection are as follows:

- **Serverless Flexibility**: Azure Functions allow us to run code without managing the infrastructure. We pay only for the resources used during execution, making it a cost-effective choice for sporadic background tasks.

- **Asynchronous Processing**: Azure Functions are well-suited for asynchronous workloads like processing messages from a Service Bus. This ensures that our application remains responsive to user requests.

- **Scalability**: Azure Functions can auto-scale based on demand, which is crucial for handling varying workloads efficiently.

This architecture selection is designed to provide a balance between performance, scalability, security, and cost-effectiveness, aligning with the project's requirements for a modern, robust, and budget-conscious application.

Remember to customize this section with specific details about your project's architecture choices and reasoning.
