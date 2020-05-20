var mysql=require('mysql');

var connection=mysql.createConnection({
    host:'localhost',
    user:'localhost',
    password:'Kdsnytl@209',
    database:'iot_app'
});
connection.connect(function(error){
   if(!!error){
     console.log(error);
   }else{
     console.log('Connected!:)');
   }
 });  
module.exports = connection; 