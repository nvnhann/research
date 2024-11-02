package com.udacity.jwdnd.course1.cloudstorage;

import io.github.bonigarcia.wdm.WebDriverManager;
import org.junit.jupiter.api.*;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.web.server.LocalServerPort;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class CredentialTests {

    @LocalServerPort
    private int port;

    private WebDriver driver;

    @BeforeAll
    static void beforeAll() {
        WebDriverManager.chromedriver().setup();
    }

    @BeforeEach
    public void beforeEach() {
        ChromeOptions options = new ChromeOptions();
        options.addArguments("--remote-allow-origins=*");
        options.addArguments("--headless");
        options.addArguments("--no-sandbox");
        this.driver = new ChromeDriver(options);
    }

    @AfterEach
    public void afterEach() {
        if (this.driver != null) {
            driver.quit();
        }
    }

    /**
     * PLEASE DO NOT DELETE THIS method.
     * Helper method for Udacity-supplied sanity checks.
     **/
    private void doMockSignUp(String firstName, String lastName, String userName, String password){
        // Create a dummy account for logging in later.

        // Visit the sign-up page.
        WebDriverWait webDriverWait = new WebDriverWait(driver, 2);
        driver.get("http://localhost:" + this.port + "/signup");
        webDriverWait.until(ExpectedConditions.titleContains("Sign Up"));

        // Fill out credentials
        webDriverWait.until(ExpectedConditions.visibilityOfElementLocated(By.id("inputFirstName")));
        WebElement inputFirstName = driver.findElement(By.id("inputFirstName"));
        inputFirstName.click();
        inputFirstName.sendKeys(firstName);

        webDriverWait.until(ExpectedConditions.visibilityOfElementLocated(By.id("inputLastName")));
        WebElement inputLastName = driver.findElement(By.id("inputLastName"));
        inputLastName.click();
        inputLastName.sendKeys(lastName);

        webDriverWait.until(ExpectedConditions.visibilityOfElementLocated(By.id("inputUsername")));
        WebElement inputUsername = driver.findElement(By.id("inputUsername"));
        inputUsername.click();
        inputUsername.sendKeys(userName);

        webDriverWait.until(ExpectedConditions.visibilityOfElementLocated(By.id("inputPassword")));
        WebElement inputPassword = driver.findElement(By.id("inputPassword"));
        inputPassword.click();
        inputPassword.sendKeys(password);

        // Attempt to sign up.
        webDriverWait.until(ExpectedConditions.visibilityOfElementLocated(By.id("buttonSignUp")));
        WebElement buttonSignUp = driver.findElement(By.id("buttonSignUp"));
        buttonSignUp.click();

		/* Check that the sign-up was successful.
		// You may have to modify the element "success-msg" and the sign-up
		// success message below depening on the rest of your code.
		*/
        Assertions.assertEquals("http://localhost:" + this.port + "/login", driver.getCurrentUrl());
    }



    /**
     * PLEASE DO NOT DELETE THIS method.
     * Helper method for Udacity-supplied sanity checks.
     **/
    private void doLogIn(String userName, String password)
    {
        // Log in to our dummy account.
        driver.get("http://localhost:" + this.port + "/login");
        WebDriverWait webDriverWait = new WebDriverWait(driver, 2);

        webDriverWait.until(ExpectedConditions.visibilityOfElementLocated(By.id("inputUsername")));
        WebElement loginUserName = driver.findElement(By.id("inputUsername"));
        loginUserName.click();
        loginUserName.sendKeys(userName);

        webDriverWait.until(ExpectedConditions.visibilityOfElementLocated(By.id("inputPassword")));
        WebElement loginPassword = driver.findElement(By.id("inputPassword"));
        loginPassword.click();
        loginPassword.sendKeys(password);

        webDriverWait.until(ExpectedConditions.visibilityOfElementLocated(By.id("login-button")));
        WebElement loginButton = driver.findElement(By.id("login-button"));
        loginButton.click();

        webDriverWait.until(ExpectedConditions.titleContains("Home"));
    }

    private void createCredential(String Url, String username, String password) {
        WebDriverWait webDriverWait = new WebDriverWait(driver, 2);
        webDriverWait.until(ExpectedConditions.visibilityOfElementLocated(By.id("nav-credentials-tab")));
        WebElement tabnote = driver.findElement(By.id("nav-credentials-tab"));
        tabnote.click();

        webDriverWait.until(ExpectedConditions.visibilityOfElementLocated(By.id("credential-creation-btn")));
        WebElement openModal = driver.findElement(By.id("credential-creation-btn"));
        openModal.click();

        webDriverWait.until(ExpectedConditions.visibilityOfElementLocated(By.id("credential-url")));
        WebElement credentialUrl = driver.findElement(By.id("credential-url"));
        credentialUrl.click();
        credentialUrl.sendKeys(Url);

        webDriverWait.until(ExpectedConditions.visibilityOfElementLocated(By.id("credential-username")));
        WebElement credentialUsername = driver.findElement(By.id("credential-username"));
        credentialUsername.click();
        credentialUsername.sendKeys(username);

        webDriverWait.until(ExpectedConditions.visibilityOfElementLocated(By.id("credential-password")));
        WebElement credentialPassword = driver.findElement(By.id("credential-password"));
        credentialPassword.click();
        credentialPassword.sendKeys(password);

        webDriverWait.until(ExpectedConditions.visibilityOfElementLocated(By.id("credentialSubmit")));
        WebElement noteSubmit = driver.findElement(By.id("credentialSubmit"));
        noteSubmit.click();

        Assertions.assertEquals("http://localhost:" + this.port + "/result?isSuccess=true#nav-credentials", driver.getCurrentUrl());
        webDriverWait.until(ExpectedConditions.visibilityOfElementLocated(By.id("success")));
        WebElement success = driver.findElement(By.id("success"));
        success.click();
    }


    @Test
    @Order(1)
    public void crudCredential() {
        doMockSignUp("URL","Test","UTCREDENTIAL","123");
        doLogIn("UTCREDENTIAL", "123");
        createCredential("/test", "UT", "UT");
        Assertions.assertTrue(driver.getPageSource().contains("test"));

        WebDriverWait webDriverWaits = new WebDriverWait(driver, 5);
        webDriverWaits.until(ExpectedConditions.visibilityOfElementLocated(By.id("credential-edit-1")));
        WebElement editButton = driver.findElement(By.id("credential-edit-1"));
        editButton.click();

        webDriverWaits.until(ExpectedConditions.visibilityOfElementLocated(By.id("credential-url")));
        WebElement credentialUrl = driver.findElement(By.id("credential-url"));
        credentialUrl.click();
        credentialUrl.sendKeys("/testupdate");

        webDriverWaits.until(ExpectedConditions.visibilityOfElementLocated(By.id("credential-username")));
        WebElement credentialUsername = driver.findElement(By.id("credential-username"));
        credentialUsername.click();
        credentialUsername.sendKeys("UT2");

        webDriverWaits.until(ExpectedConditions.visibilityOfElementLocated(By.id("credential-password")));
        WebElement credentialPassword = driver.findElement(By.id("credential-password"));
        credentialPassword.click();
        credentialPassword.sendKeys("UT2");

        webDriverWaits.until(ExpectedConditions.visibilityOfElementLocated(By.id("credentialSubmit")));
        WebElement noteSubmit = driver.findElement(By.id("credentialSubmit"));
        noteSubmit.click();

        Assertions.assertEquals("http://localhost:" + this.port + "/result?isSuccess=true#nav-credentials", driver.getCurrentUrl());
        webDriverWaits.until(ExpectedConditions.visibilityOfElementLocated(By.id("success")));
        WebElement success = driver.findElement(By.id("success"));
        success.click();
        Assertions.assertTrue(driver.getPageSource().contains("testupdate"));

        webDriverWaits.until(ExpectedConditions.visibilityOfElementLocated(By.id("credential-delete-1")));
        WebElement deleteButton = driver.findElement(By.id("credential-delete-1"));
        deleteButton.click();
        Assertions.assertEquals("http://localhost:" + this.port + "/result?isSuccess=true#nav-credentials", driver.getCurrentUrl());
        webDriverWaits.until(ExpectedConditions.visibilityOfElementLocated(By.id("success")));
        WebElement deleteBtn = driver.findElement(By.id("success"));
        deleteBtn.click();
        Assertions.assertFalse(driver.getPageSource().contains("testupdate"));
    }
}