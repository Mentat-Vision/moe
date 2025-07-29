#include "esp_camera.h"
#include <WiFi.h>
#include <Preferences.h>
#include <DNSServer.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "esp_http_server.h"
#include "esp_task_wdt.h"

// 🔐 WiFi provisioning
Preferences preferences;
DNSServer dnsServer;
const char* ap_ssid = "Mentat";
const char* ap_password = "12345678";
bool wifi_provisioned = false;
String saved_ssid = "";
String saved_password = "";

// 🌐 Network configuration (using DHCP)

// 📷 AI Thinker ESP32-CAM pin mapping
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Flash LED on AI Thinker ESP32-CAM (GPIO 4)
#define FLASH_LED         4

// Reset button (we'll use GPIO 0 - BOOT button on ESP32-CAM)
#define RESET_BUTTON      0

// 📡 MJPEG streaming handler
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  char *part_buf[64];
  static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=frame";
  static const char* _STREAM_BOUNDARY = "\r\n--frame\r\n";
  static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

  httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("❌ Camera capture failed");
      return ESP_FAIL;
    }

    size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, fb->len);

    // Exit cleanly if client disconnects
    if (httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY)) != ESP_OK ||
        httpd_resp_send_chunk(req, (const char *)part_buf, hlen) != ESP_OK ||
        httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len) != ESP_OK) {
      esp_camera_fb_return(fb);
      Serial.println("🔌 Client disconnected, stream ended");
      break;
    }

    esp_camera_fb_return(fb);
    // Optional: delay(10); // add to reduce load if needed
  }

  return ESP_OK;
}

// 📡 HTTP route registration
static const httpd_uri_t stream_uri = {
  .uri       = "/stream",
  .method    = HTTP_GET,
  .handler   = stream_handler,
  .user_ctx  = NULL
};

// 🌐 WiFi setup page handler
static esp_err_t setup_handler(httpd_req_t *req) {
  const char* html = R"rawliteral(
<!DOCTYPE html>
<html><head><title>Mentat - WiFi Camera Setup</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:'Segoe UI',Arial,sans-serif;padding:20px;background:#000;color:#fff;margin:0;}
.container{max-width:400px;margin:0 auto;background:#111;padding:30px;border:2px solid #ff0000;box-shadow:0 0 20px rgba(255,0,0,0.3);}
h1{color:#ff0000;text-align:center;margin-bottom:10px;font-size:28px;font-weight:bold;text-transform:uppercase;letter-spacing:2px;}
.subtitle{color:#ccc;text-align:center;margin-bottom:30px;font-size:14px;}
input{width:100%;padding:15px;margin:10px 0;border:2px solid #333;background:#222;color:#fff;box-sizing:border-box;font-size:16px;-webkit-appearance:none;-moz-appearance:none;appearance:none;}
input:focus{outline:none;border-color:#ff0000;box-shadow:0 0 5px rgba(255,0,0,0.5);}
button{width:100%;background:#ff0000;color:#000;padding:15px;border:none;cursor:pointer;font-size:16px;font-weight:bold;text-transform:uppercase;letter-spacing:1px;transition:all 0.3s;-webkit-appearance:none;-moz-appearance:none;appearance:none;}
button:hover{background:#cc0000;box-shadow:0 0 10px rgba(255,0,0,0.7);}
.info{background:#333;padding:15px;margin-bottom:20px;border-left:4px solid #ff0000;color:#ccc;}
</style></head>
<body>
<div class="container">
<h1>Mentat</h1>
<div class="subtitle">WiFi Camera Setup</div>
<div class="info">Connect your camera to WiFi network to begin surveillance operations.</div>
<form action="/save" method="POST">
<input type="text" name="ssid" placeholder="WiFi Network Name (SSID)" required>
<input type="text" name="password" placeholder="WiFi Password" required>
<button type="submit">Initialize Connection</button>
</form>
</div>
</body></html>
)rawliteral";
  
  httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
  return ESP_OK;
}

// 💾 Save WiFi credentials handler
static esp_err_t save_handler(httpd_req_t *req) {
  char content[200];
  size_t recv_size = (req->content_len < sizeof(content)) ? req->content_len : sizeof(content);
  
  int ret = httpd_req_recv(req, content, recv_size);
  if (ret <= 0) {
    if (ret == HTTPD_SOCK_ERR_TIMEOUT) {
      httpd_resp_send_408(req);
    }
    return ESP_FAIL;
  }
  
  content[ret] = '\0';
  
  // Parse form data (simple parsing for ssid=...&password=...)
  char* ssid_start = strstr(content, "ssid=");
  char* password_start = strstr(content, "&password=");
  
  if (ssid_start && password_start) {
    ssid_start += 5; // Skip "ssid="
    *password_start = '\0'; // Terminate SSID
    password_start += 10; // Skip "&password="
    
    // URL decode (basic)
    String new_ssid = String(ssid_start);
    String new_password = String(password_start);
    new_ssid.replace("+", " ");
    new_password.replace("+", " ");
    
    // Save to preferences
    preferences.begin("wifi", false);
    preferences.putString("ssid", new_ssid);
    preferences.putString("password", new_password);
    preferences.end();
    
    Serial.println("✅ WiFi credentials saved, restarting...");
    
    const char* response = R"rawliteral(
<!DOCTYPE html><html><head><title>Connecting...</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:'Segoe UI',Arial,sans-serif;padding:20px;background:#000;color:#fff;margin:0;}
.container{max-width:400px;margin:0 auto;background:#111;padding:30px;border:2px solid #ff0000;box-shadow:0 0 20px rgba(255,0,0,0.3);text-align:center;}
h1{color:#ff0000;font-size:24px;margin-bottom:20px;}
.status{color:#ccc;margin-bottom:20px;}
.led{display:inline-block;width:20px;height:20px;border-radius:50%;margin:0 5px;background:#333;}
.led.on{background:#ff0000;box-shadow:0 0 10px rgba(255,0,0,0.7);}
</style></head>
<body>
<div class="container">
<h1>Connecting to WiFi...</h1>
<div class="status">Watch the LED on your device:</div>
<div style="margin:20px 0;">
<div class="led on"></div> <span>Fast blinking = Connecting</span><br><br>
<div class="led on"></div><div class="led on"></div> <span>Double flash = Connected & Ready</span>
</div>
<p>This page will close automatically.</p>
</div>
<script>setTimeout(function(){window.close();}, 3000);</script>
</body></html>
)rawliteral";
    httpd_resp_send(req, response, HTTPD_RESP_USE_STRLEN);
    
    // Signal that we're starting WiFi connection with fast LED blinking
    for (int i = 0; i < 6; i++) {
      digitalWrite(FLASH_LED, HIGH);
      delay(100);
      digitalWrite(FLASH_LED, LOW);
      delay(100);
    }
    
    delay(500);
    ESP.restart();
  }
  
  return ESP_OK;
}

// 🔄 Reset endpoint handler
static esp_err_t reset_handler(httpd_req_t *req) {
  const char* response = R"rawliteral(
<!DOCTYPE html><html><head><title>Factory Reset</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:'Segoe UI',Arial,sans-serif;padding:20px;background:#000;color:#fff;margin:0;}
.container{max-width:400px;margin:0 auto;background:#111;padding:30px;border:2px solid #ff0000;box-shadow:0 0 20px rgba(255,0,0,0.3);text-align:center;}
h1{color:#ff0000;font-size:24px;margin-bottom:20px;}
.status{color:#ccc;margin-bottom:20px;}
.led{display:inline-block;width:20px;height:20px;border-radius:50%;margin:0 5px;background:#ff0000;box-shadow:0 0 10px rgba(255,0,0,0.7);}
</style></head>
<body>
<div class="container">
<h1>Factory Reset Complete</h1>
<div class="status">WiFi credentials have been cleared.</div>
<div style="margin:20px 0;">
<div class="led"></div><div class="led"></div><div class="led"></div><div class="led"></div><div class="led"></div>
<br><span>5 LED blinks = Reset confirmed</span>
</div>
<p>Device is restarting in setup mode...</p>
</div>
<script>setTimeout(function(){window.close();}, 3000);</script>
</body></html>
)rawliteral";

  httpd_resp_send(req, response, HTTPD_RESP_USE_STRLEN);
  
  // Trigger factory reset after sending response
  delay(1000);
  factoryReset();
  
  return ESP_OK;
}

// Captive portal - catch all requests
static esp_err_t catchall_handler(httpd_req_t *req) {
  // Redirect all requests to the setup page
  return setup_handler(req);
}

static const httpd_uri_t setup_uri = {
  .uri       = "/",
  .method    = HTTP_GET,
  .handler   = setup_handler,
  .user_ctx  = NULL
};

static const httpd_uri_t catchall_uri = {
  .uri       = "/*",
  .method    = HTTP_GET,
  .handler   = catchall_handler,
  .user_ctx  = NULL
};

static const httpd_uri_t save_uri = {
  .uri       = "/save",
  .method    = HTTP_POST,
  .handler   = save_handler,
  .user_ctx  = NULL
};

static const httpd_uri_t reset_uri = {
  .uri       = "/reset",
  .method    = HTTP_GET,
  .handler   = reset_handler,
  .user_ctx  = NULL
};

// 🚀 Start the HTTP server (setup mode)
void startSetupServer() {
  Serial.println("🚀 Starting setup web server...");
  
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.max_open_sockets = 7;
  config.task_priority = tskIDLE_PRIORITY + 5;

  httpd_handle_t server = NULL;
  esp_err_t err = httpd_start(&server, &config);
  
  if (err == ESP_OK) {
    Serial.println("📊 HTTP server created successfully");
    
    // Register handlers
    esp_err_t err1 = httpd_register_uri_handler(server, &setup_uri);
    esp_err_t err2 = httpd_register_uri_handler(server, &save_uri);
    esp_err_t err3 = httpd_register_uri_handler(server, &reset_uri);
    esp_err_t err4 = httpd_register_uri_handler(server, &catchall_uri);
    
    if (err1 == ESP_OK && err2 == ESP_OK && err3 == ESP_OK && err4 == ESP_OK) {
      Serial.println("✅ Setup server started on port 80");
      Serial.println("🔗 Connect to Mentat WiFi and go to http://192.168.4.1");
      Serial.println("🔄 Factory reset: http://192.168.4.1/reset");
      Serial.println("🌐 Any URL should redirect to setup page");
    } else {
      Serial.println("❌ Failed to register URI handlers");
    }
  } else {
    Serial.print("❌ Failed to start HTTP server, error: ");
    Serial.println(err);
  }
}

// 🚀 Start the HTTP server (camera mode)
void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;  // use port 80 for better compatibility

  httpd_handle_t server = NULL;
  if (httpd_start(&server, &config) == ESP_OK) {
    httpd_register_uri_handler(server, &stream_uri);
    httpd_register_uri_handler(server, &reset_uri); // Allow reset even in camera mode
    Serial.println("✅ HTTP server started on port 80");
    Serial.println("🔄 Factory reset available at: http://[device-ip]/reset");
  } else {
    Serial.println("❌ Failed to start HTTP server");
  }
}

// 📸 Initialize camera
void setupCamera() {
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

  config.frame_size = FRAMESIZE_QVGA;      // Options: QVGA/VGA/SVGA
  config.jpeg_quality = 12;                // Lower = better quality, higher = smaller size
  config.fb_count = 2;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("❌ Camera init failed with error 0x%x", err);
    return;
  }

  // 🔄 Apply rotation (vflip + hmirror = ~90° clockwise)
  sensor_t *s = esp_camera_sensor_get();
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
  Serial.println("✅ Camera initialized and rotated");
}

// 🔄 Factory reset function
void factoryReset() {
  Serial.println("🔄 Performing factory reset...");
  
  // Clear WiFi credentials
  preferences.begin("wifi", false);
  preferences.clear();
  preferences.end();
  
  // LED confirmation pattern: 5 fast blinks
  for (int i = 0; i < 5; i++) {
    digitalWrite(FLASH_LED, HIGH);
    delay(150);
    digitalWrite(FLASH_LED, LOW);
    delay(150);
  }
  
  Serial.println("✅ Factory reset completed - restarting in setup mode");
  delay(1000);
  ESP.restart();
}

// 🔘 Check for reset button during boot
bool checkResetButton() {
  pinMode(RESET_BUTTON, INPUT_PULLUP);
  
  if (digitalRead(RESET_BUTTON) == LOW) {
    Serial.println("🔘 Reset button detected, hold for 5 seconds to factory reset...");
    
    // Show countdown with LED
    for (int i = 0; i < 50; i++) { // 5 seconds = 50 * 100ms
      if (digitalRead(RESET_BUTTON) == HIGH) {
        Serial.println("❌ Reset button released, continuing normal boot");
        return false;
      }
      
      // Blink LED during countdown
      digitalWrite(FLASH_LED, (i % 10 < 5) ? HIGH : LOW);
      delay(100);
    }
    
    // Button held for 5 seconds
    digitalWrite(FLASH_LED, LOW);
    Serial.println("✅ Factory reset triggered!");
    return true;
  }
  
  return false;
}

// 💬 Handle serial commands
void handleSerialCommands() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    command.toUpperCase();
    
    if (command == "RESET" || command == "FACTORY_RESET") {
      Serial.println("🔄 Factory reset command received via serial");
      factoryReset();
    } else if (command == "STATUS") {
      Serial.println("📊 Device Status:");
      Serial.print("WiFi: ");
      Serial.println(wifi_provisioned ? "Connected" : "Setup Mode");
      if (wifi_provisioned) {
        Serial.print("IP: ");
        Serial.println(WiFi.localIP());
      }
      Serial.print("SSID: ");
      Serial.println(saved_ssid.length() > 0 ? saved_ssid : "None");
    } else if (command == "HELP") {
      Serial.println("📋 Available Commands:");
      Serial.println("RESET - Factory reset device");
      Serial.println("STATUS - Show device status");
      Serial.println("HELP - Show this help");
    } else if (command.length() > 0) {
      Serial.println("❓ Unknown command. Type HELP for available commands.");
    }
  }
}

// 🔧 Setup everything
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // disable brownout
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  delay(2000); // Give time for serial to initialize

  // Initialize flash LED
  pinMode(FLASH_LED, OUTPUT);
  digitalWrite(FLASH_LED, LOW); // Turn off initially

  // Check for factory reset button during boot
  if (checkResetButton()) {
    factoryReset();
    return; // This line won't be reached due to ESP.restart()
  }

  // Enable watchdog timer (30 seconds)
  esp_task_wdt_config_t twdt_config = {
    .timeout_ms = 30000,
    .idle_core_mask = (1 << portNUM_PROCESSORS) - 1,
    .trigger_panic = true,
  };
  esp_task_wdt_init(&twdt_config);
  esp_task_wdt_add(NULL);

  // Check for saved WiFi credentials
  preferences.begin("wifi", true);
  saved_ssid = preferences.getString("ssid", "");
  saved_password = preferences.getString("password", "");
  preferences.end();

  if (saved_ssid.length() > 0) {
    // Try to connect with saved credentials
    Serial.println("🔐 Found saved WiFi credentials, attempting connection...");
    WiFi.begin(saved_ssid.c_str(), saved_password.c_str());
    
    // Fast LED blinking during connection attempt
    unsigned long startAttemptTime = millis();
    bool ledState = false;
    while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < 15000) {
      // Fast blink every 200ms during WiFi connection
      static unsigned long lastBlink = 0;
      if (millis() - lastBlink > 200) {
        ledState = !ledState;
        digitalWrite(FLASH_LED, ledState ? HIGH : LOW);
        lastBlink = millis();
      }
      
      delay(50);
      Serial.print(".");
      esp_task_wdt_reset();
    }
    
    // Turn off LED after connection attempt
    digitalWrite(FLASH_LED, LOW);

    if (WiFi.status() == WL_CONNECTED) {
      wifi_provisioned = true;
      Serial.println("\n✅ WiFi connected with saved credentials");
      Serial.print("🌐 IP address: ");
      Serial.println(WiFi.localIP());
      
      setupCamera();
      startCameraServer();
      
      // Flash LED twice to indicate server is ready
      for (int i = 0; i < 2; i++) {
        digitalWrite(FLASH_LED, HIGH);
        delay(200);
        digitalWrite(FLASH_LED, LOW);
        delay(200);
      }
      
      Serial.println("📸 MJPEG stream ready");
      Serial.printf("🔗 Stream URL: http://%s/stream\n", WiFi.localIP().toString().c_str());
      return;
    } else {
      Serial.println("\n❌ Saved credentials failed, starting setup mode...");
    }
  } else {
    Serial.println("🔧 No saved WiFi credentials, starting setup mode...");
  }

  // Start Access Point for WiFi setup
  WiFi.mode(WIFI_AP);
  
  // Configure AP settings (up to 4 devices can connect by default)
  IPAddress local_IP(192, 168, 4, 1);
  IPAddress gateway(192, 168, 4, 1);
  IPAddress subnet(255, 255, 255, 0);
  
  WiFi.softAPConfig(local_IP, gateway, subnet);
  
  bool ap_started = WiFi.softAP(ap_ssid, ap_password, 1, 0, 4); // channel 1, hidden=false, max 4 connections
  
  if (ap_started) {
    Serial.println("📡 Access Point started successfully");
    Serial.print("🔗 SSID: ");
    Serial.println(ap_ssid);
    Serial.print("🔑 Password: ");
    Serial.println(ap_password);
    Serial.print("🌐 AP IP: ");
    Serial.println(WiFi.softAPIP());
    Serial.print("📱 Max connections: 4");
    Serial.println();
    
    // Start DNS server for captive portal
    dnsServer.start(53, "*", local_IP);
    
    startSetupServer();
  } else {
    Serial.println("❌ Failed to start Access Point");
    ESP.restart();
  }
  
  // Flash LED once every 2 seconds to indicate setup mode
  Serial.println("⚙️ Setup mode active - connect to Mentat WiFi");
}

// 🔁 Monitor connection and reset watchdog
void loop() {
  // Reset watchdog timer
  esp_task_wdt_reset();
  
  // Handle serial commands
  handleSerialCommands();
  
  if (wifi_provisioned) {
    // Normal operation mode - monitor WiFi connection
    static unsigned long lastCheck = 0;
    if (millis() - lastCheck > 30000) {
      lastCheck = millis();
      
      if (WiFi.status() != WL_CONNECTED) {
        Serial.println("⚠️ WiFi disconnected - attempting reconnect...");
        WiFi.reconnect();
        
        // Wait up to 10 seconds for reconnection
        unsigned long reconnectStart = millis();
        while (WiFi.status() != WL_CONNECTED && millis() - reconnectStart < 10000) {
          delay(500);
          Serial.print(".");
          esp_task_wdt_reset();
        }
        
        if (WiFi.status() == WL_CONNECTED) {
          Serial.println("\n✅ WiFi reconnected");
        } else {
          Serial.println("\n❌ WiFi reconnection failed - restarting...");
          ESP.restart();
        }
      }
    }
    delay(1000);
  } else {
    // Setup mode - handle DNS requests and flash LED
    dnsServer.processNextRequest();
    
    // Flash LED slowly to indicate setup mode
    static unsigned long lastFlash = 0;
    if (millis() - lastFlash > 2000) {
      lastFlash = millis();
      digitalWrite(FLASH_LED, HIGH);
      delay(100);
      digitalWrite(FLASH_LED, LOW);
    }
    
    delay(100);
  }
}
