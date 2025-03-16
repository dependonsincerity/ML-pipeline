pipeline {
    agent any

    stages {
        stage('git_clone') {
            steps {
                git branch: 'master', url: 'https://github.com/dependonsincerity/ML-pipeline.git' // URL репозитория
                dir("lab1") {
                    sh "python3 -m venv ./venv"
                    sh ". ./venv/bin/activate"
                    sh "venv/bin/python3 -m pip install scikit-learn numpy pandas"
                }
            }
        }
        stage('data_creation') {
            steps {
                dir('lab1') {
                    sh "venv/bin/python3 data_creation.py" // скрипт для создания данных
                }
            }
        }
        stage('data_preprocessing') {
            steps {
                dir('lab1') {
                    sh "venv/bin/python3 data_preprocessing.py" // скрипт для предобработки данных
                }
            }
        }
        stage('model_preparation') {
            steps {
                dir('lab1') {
                    sh "venv/bin/python3 model_preparation.py" // скрипт для подготовки модели
                }
            }
        }
        stage('model_testing') {
            steps {
                dir('lab1') {
                    sh "venv/bin/python3 model_testing.py" // скрипт для тестирования модели
                }
            }
        }
    }
}