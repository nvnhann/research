pipeline {
    agent any

    tools {
        maven 'Maven'
        jdk 'JDK 11'
    }

    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/nvnhann/eCommerce-Application.git'
            }
        }
        stage('Build') {
            steps {
                dir('starter_code') {
                    sh 'mvn clean package'
                }
            }
        }
        stage('Test') {
            steps {
                dir('starter_code') {
                    sh 'mvn test'
                }
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed successfully.'
        }
        failure {
            echo 'Pipeline failed.'
        }
    }
}