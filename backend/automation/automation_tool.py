from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import unittest

class WebTestAutomation:
    def __init__(self, driver_path, base_url):
        """
        Initialize the WebTestAutomation with the given parameters.
        :param driver_path: Path to the Selenium WebDriver executable.
        :param base_url: URL of the application to test.
        """
        self.driver_path = driver_path
        self.base_url = base_url
        self.driver = None

    def start_browser(self):
        """Start the Selenium WebDriver and open the base URL."""
        # Set up the Selenium WebDriver (you can use ChromeDriver, FirefoxDriver, etc.)
        self.driver = webdriver.Chrome(executable_path=self.driver_path)
        self.driver.get(self.base_url)

    def login_test(self, username, password):
        """Test logging in with the given username and password."""
        try:
            # Locate username and password fields, then input test data
            username_field = self.driver.find_element(By.ID, "username")
            password_field = self.driver.find_element(By.ID, "password")
            login_button = self.driver.find_element(By.ID, "login_button")

            username_field.send_keys(username)
            password_field.send_keys(password)
            login_button.click()

            # Add assertion to confirm login was successful
            time.sleep(2)  # Wait for login to complete
            assert "Dashboard" in self.driver.title, "Login failed!"
            print("Login test passed!")
        except Exception as e:
            print(f"Error during login test: {e}")

    def logout_test(self):
        """Test logging out from the application."""
        try:
            logout_button = self.driver.find_element(By.ID, "logout_button")
            logout_button.click()

            # Add assertion to confirm logout was successful
            time.sleep(2)
            assert "Login" in self.driver.title, "Logout failed!"
            print("Logout test passed!")
        except Exception as e:
            print(f"Error during logout test: {e}")

    def run_tests(self):
        """Run the full suite of tests."""
        self.start_browser()
        self.login_test("test_user", "test_password")
        self.logout_test()

    def close_browser(self):
        """Close the browser after testing."""
        if self.driver:
            self.driver.quit()

# Example Usage
if __name__ == "__main__":
    # Provide path to your webdriver (e.g., 'chromedriver.exe' for Chrome)
    driver_path = "C:\Users\ibrahim.fadhili\Downloads\chrome-win64\chrome-win64\chrome.exe"
    base_url = "http://localhost:5000"  # Replace with your application's URL

    # Initialize the test automation class
    automation = WebTestAutomation(driver_path, base_url)

    try:
        # Run all tests
        automation.run_tests()
    finally:
        # Ensure that the browser is closed after tests are complete
        automation.close_browser()
