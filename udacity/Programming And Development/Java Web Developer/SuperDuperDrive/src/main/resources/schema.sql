CREATE TABLE IF NOT EXISTS USERS (
                                     userid INT AUTO_INCREMENT PRIMARY KEY,
                                     username VARCHAR(20) UNIQUE NOT NULL,
    salt VARCHAR(64) NOT NULL,
    password VARCHAR(512) NOT NULL,
    firstname VARCHAR(20) NOT NULL,
    lastname VARCHAR(20) NOT NULL
    );

CREATE TABLE IF NOT EXISTS NOTES (
                                     noteid INT AUTO_INCREMENT PRIMARY KEY,
                                     notetitle VARCHAR(100) NOT NULL,
    notedescription VARCHAR(1000) NOT NULL,
    userid INT NOT NULL,
    FOREIGN KEY (userid) REFERENCES USERS(userid)
    );

CREATE TABLE IF NOT EXISTS FILES (
                                     fileid INT AUTO_INCREMENT PRIMARY KEY,
                                     filename VARCHAR(255) NOT NULL,
    contenttype VARCHAR(100) NOT NULL,
    filesize VARCHAR(100) NOT NULL,
    userid INT NOT NULL,
    filedata LONGBLOB NOT NULL,
    FOREIGN KEY (userid) REFERENCES USERS(userid)
    );

CREATE TABLE IF NOT EXISTS CREDENTIALS (
                                           credentialid INT AUTO_INCREMENT PRIMARY KEY,
                                           url VARCHAR(100) NOT NULL,
    username VARCHAR(30) NOT NULL,
    `key` VARCHAR(128) NOT NULL,
    password VARCHAR(128) NOT NULL,
    userid INT NOT NULL,
    FOREIGN KEY (userid) REFERENCES USERS(userid)
    );