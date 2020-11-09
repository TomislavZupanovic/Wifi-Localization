#include <SPI.h>
#include <WiFiNINA.h>

void setup() {

  //Initialize serial and wait for port to open:

  Serial.begin(9600);

  while (!Serial) {

    ; // wait for serial port to connect. Needed for native USB port only

  }

  // check for the WiFi module:

  if (WiFi.status() == WL_NO_MODULE) {

    Serial.println("Communication with WiFi module failed!");

    // don't continue

    while (true);

  }
}

void loop() {

  // scan for existing networks:

  listNetworks();

  delay(100);
}

void listNetworks() {

  // scan for nearby networks:

  int numSsid = WiFi.scanNetworks();

  if (numSsid == -1) {

    Serial.println("Couldn't get a wifi connection");

    while (true);

  }

  // print the list of networks seen:

  // print the network number and name for each network found:

  for (int thisNet = 0; thisNet < numSsid; thisNet++) {

    Serial.print(WiFi.SSID(thisNet));
    Serial.print(",");
    Serial.print(WiFi.RSSI(thisNet));
    Serial.print(",");
    Serial.print("A506");
    Serial.print("\n");

    

  }
}
