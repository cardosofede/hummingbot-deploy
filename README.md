# hummingbot-deploy

Welcome to the Hummingbot Deploy project. This guide will walk you through the steps to deploy multiple trading bots using a centralized dashboard and a service backend.

## Prerequisites

- Docker must be installed on your machine. If you do not have Docker installed, you can download and install it from [Docker's official site](https://www.docker.com/products/docker-desktop).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cardosofede/hummingbot-deploy.git
   cd hummingbot-deploy
   ```

2. **Configure the .m file:**
   - Locate the `.env` file in the repository.
   - Edit the `.env` file to update the bots path to reflect the path where your project is located. This path is crucial as it maps the backend API functionalities to your host machine.

3. **Pull the Docker image:**
   ```bash
   docker pull dardonacci/hummingbot
   ```

4. **API Keys and Credentials:**
   - Create API keys with a hummingbot instance, the password of the instance should be "a" to work with the current setup, in future versions this will be all handled by the dashboard.
   - Navigate to the `bots_credentials/master_account` directory.
   - Place your YAML files with the required API keys and credentials in this directory.

## Running the Application

1. **Start the application:**
   ```bash
   docker-compose up -d
   ```

2. **Access the dashboard:**
   - Open your web browser and go to `localhost:8501`.

3. **Create a config for D-Man Maker V2**
   - Go to the tab D-Man Maker V2 and create a new configuration. Soon will be released a video explaining how the strategy works.

4. **Deploy the configuration**
   - Go to the Deploy tab, select a name for your bot, the image dardonacci/hummingbot:latest and the configuration you just created.
   - Press the button to create a new instance.

5. **Check the status of the bot**
   - Go to the Instances tab and check the status of the bot.
     - If it's not available is because the bot is starting, wait a few seconds and refresh the page.
     - If it's running, you can check the performance of it in the graph, refresh to see the latest data.
     - If it's stopped, probably the bot had an error, you can check the logs in the container to understand what happened.

5. **[Optional] Check the Backend API**
   -  Open your web browser and go to `localhost:8000/docs`.

## Dashboard Functionalities

- **D-Man v2 Configurations:**
  - Create and select configurations for the Daemon v2 strategy.
  - Deploy the selected configurations.

- **Bot Management:**
  - Visualize bot performance in real-time.
  - Stop and archive running bots.

## Tutorial

To get started with deploying your first bot, follow these step-by-step instructions:

1. **Prepare your bot configurations:**
   - Ensure you have the correct YAML configuration files in your `master_account` folder.

2. **Deploy a bot:**
   - Use the dashboard UI to select and deploy your configurations.

3. **Monitor and Manage:**
   - Track bot performance and make adjustments as needed through the dashboard.