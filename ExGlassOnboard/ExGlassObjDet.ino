#include <WiFi.h>
#include <esp_camera.h>

#define CAMERA_MODEL_AI_THINKER  // Uncomment for AI Thinker module
#include "camera_pins.h"

// WiFi credentials
const char* ssid = "flambo";
const char* password = "ceaHppsPd6gAep7x";

WiFiServer server(80);

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  // Camera configuration
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // Frame settings: Adjust resolution for performance
  config.frame_size = FRAMESIZE_VGA;  
  config.jpeg_quality = 10;  
  config.fb_count = 2;

  // Initialize camera
  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed");
    return;
  }

  // Start the server
  server.begin();
  Serial.print("MJPEG Stream ready! Go to: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/stream");
}

void loop() {
  WiFiClient client = server.available();
  if (!client) {
    delay(1);
    return;
  }
  
  // Wait for the client to send the request
  String req = client.readStringUntil('\r');
  client.flush();
  
  if (req.indexOf("GET /stream") != -1) {
    // Send HTTP header for MJPEG stream
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: multipart/x-mixed-replace; boundary=frame");
    client.println("Cache-Control: no-cache");
    client.println();
    
    while (client.connected()) {
      camera_fb_t * fb = esp_camera_fb_get();
      if (!fb) {
        Serial.println("Camera capture failed");
        break;
      }
      
      // Send MJPEG frame boundary and header
      client.println("--frame");
      client.println("Content-Type: image/jpeg");
      client.print("Content-Length: ");
      client.println(fb->len);
      client.println();
      
      // Write the frame data
      client.write(fb->buf, fb->len);
      client.println();
      
      esp_camera_fb_return(fb);
      
      // Optional: Delay between frames (adjust for your frame rate)
      delay(30);
    }
  } else {
    // For other requests, send a simple response or error.
    client.println("HTTP/1.1 404 Not Found");
    client.println("Content-Type: text/plain");
    client.println();
    client.println("Not Found");
  }
  delay(1);
}
