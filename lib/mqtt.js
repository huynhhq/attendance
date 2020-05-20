var mqtt = require('mqtt');
var connection  = require('../lib/db');
var LibLogger = require('../lib/log');
var libLogger = new LibLogger().getInstance();

class MqttHandler {

  constructor() {
    this.mqttClient = null;
    this.host = 'mqtt://192.168.71.21:1883';
  }

  connect() {
    this.mqttClient  = mqtt.connect( this.host );
    
    // Mqtt error calback
    this.mqttClient.on('error', (err) => {
      console.log(err);
      this.mqttClient.end();
    });

    // Connection callback
    this.mqttClient.on('connect', () => {
      console.log(`mqtt client connected`);
    });

    // mqtt subscriptions
    this.mqttClient.subscribe('mytopic', {qos: 0});
    this.mqttClient.subscribe('addFinger');
    this.mqttClient.subscribe('scanFinger');

    // When a message arrives, console.log it
    this.mqttClient.on('message', function (topic, message) {
      var currunt_time = new Date(); 
      console.log( 'TOPIC: ', topic.toString() );
      console.log( 'MESSAGE: ', message.toString() );
      console.log( 'TIME: ', currunt_time.toString() );

      if( topic == 'addFinger' )
      {
        let message_str = message.toString();
        let userCode = message_str.substring( 0, message_str.indexOf('_') );
        let finger = message_str.substring( message_str.indexOf('_') + 1 );
        connection.query('SELECT * FROM Users WHERE code = ' + userCode, function(err, rows, fields) {
          if(err) throw err
  
          if (rows.length <= 0) {
            libLogger.log( 'ERROR: Users not found with code = ' + userCode );              
          }
          else { 
            let user_id = rows[0].id;             
            var finger_data = {
              user_id: user_id,
              code: finger,
              status: 'active',
              created_at: currunt_time
            };          
            connection.query('INSERT INTO Fingers SET ?', finger_data, function(err, result) {                
              if (err) {
                libLogger.log( 'ERROR: Insert Fingers with user code ' + userCode +' get err = ' + err );                            
              } else {  
                libLogger.log( 'SUCCESS: Fingers not found with user code = ' + userCode );                               
              }
            });
          }            
        });
      }
      
      if( topic == 'scanFinger' )
      {
        let finger = message.toString();

        connection.query('SELECT * FROM Fingers WHERE code = ' + finger, function(err, rows, fields) {
          if(err) throw err

          if (rows.length <= 0) {
            libLogger.log( 'ERROR: Fingers not found with code = ' + finger );              
          }
          else{
            let finger_id = rows[0].id;
            let user_id = rows[0].user_id;
            var attendance_data = {
              user_id: user_id,
              finger_id: finger_id,
              status: 'active',
              created_at: currunt_time
            };
            connection.query('INSERT INTO Attendances SET ?', attendance_data, function(err, result) {                
              if (err) {
                libLogger.log( 'ERROR: Insert Attendances with finger = ' + finger +' get err = ' + err );                                      
              } else {  
                libLogger.log( 'SUCCESS: Insert Attendances with finger = ' + finger +' get err = ' + err );                                      
              }
            })
          }
        });
      }
    });

    this.mqttClient.on('close', () => {
      console.log(`mqtt client disconnected`);
    });  
  }

  sendMessage( topic, message ) {
    console.log('My TOPIC: ' + topic);    
    this.mqttClient.publish( topic, message);
  }  
}

class Singleton {

  constructor() {
    if (!Singleton.instance) {
        Singleton.instance = new MqttHandler();
    }
  }

  getInstance() {
    return Singleton.instance;
  }
}

module.exports = Singleton;



// client.on('connect', function () {
//   client.subscribe('abba');
//   client.publish('presence', 'Hello mqtt');  
// })

// client.on('message', function (topic, message) {  
//   console.log(topic.toString());
//   console.log(message.toString());
//   client.end()
// })

// function intervalFunc() {
//   console.log('Publish topic presence !');
//   client.publish('presence', 'how are you?');
// }