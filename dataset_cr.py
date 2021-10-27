#Imports Packages
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time 
#Opens up web driver and goes to Google Images
driver = webdriver.Chrome('C:/web_scrap/chromedriver.exe')
driver.get('https://images.google.com/')
box = driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input')
charec='ezio'
box.send_keys(charec)
box.send_keys(Keys.ENTER)

#driver.find_element_by_xpath('//*[@id="i6"]/div[1]/span/span/div[12]/a').click()
#time.sleep(2)

#Will keep scrolling down the webpage until it cannot scroll no more
last_height = driver.execute_script('return document.body.scrollHeight')
while True:
    driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    time.sleep(2)
    new_height = driver.execute_script('return document.body.scrollHeight')
    try:
        driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').click()
        time.sleep(2)
    except:
        pass
    if new_height == last_height:
        break
    last_height = new_height



for i in range(1, 150):
    try:
        driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').screenshot('D:/Neuroscience_dataset/dataset/'+charec+'/'+charec+'_'+str(i)+'.png')
    except:
        pass