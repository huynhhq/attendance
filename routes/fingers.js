var express = require('express');
var router = express.Router();
var connection  = require('../lib/db');
var LibLogger = require('../lib/log');
var libLogger = new LibLogger().getInstance();

router.get('/', function(req, res, next) {
      
    connection.query('SELECT * FROM Fingers ORDER BY id desc',function(err,rows)     {
        if(err){
            req.flash('error', err); 
            res.render('fingers',{page_title:"Fingers - Node.js",data:''});   
        }else{     
            for (let index = 0; index < rows.length; index++) {
                const current_finger = rows[index];
                connection.query('SELECT * FROM Users WHERE id = ' + current_finger.user_id, function(err, userRows, fields) {
                    if(err) throw err

                    if (userRows.length <= 0) {
                        userRows[index].name = 'Đang cập nhật';
                        userRows[index].user_code = 'Đang cập nhật';                        
                    }
                    else { 
                        rows[index].name = userRows[0].name;
                        rows[index].user_code = userRows[0].code;                                                                      
                    } 
                });
            }
            
            res.render('fingers',{page_title:"Fingers - Node.js",data:rows});
        }                            
    });        
});

router.get('/get/list', function(req, res, next) {
      
    connection.query('SELECT * FROM Fingers ORDER BY id desc',function(err,rows)     {
        if(err){
            res.setHeader('Content-Type', 'application/json');
            res.send(JSON.stringify({ success: false, message: 'Get Finger List Error' })); 
        }else{  
            var fingerArr = [];    
            for (let index = 0; index < rows.length; index++) {
                const current_finger = rows[index];
                let userFinger = {};
                userFinger['id'] = current_finger.id;
                userFinger['code'] = current_finger.code;
                userFinger['user_id'] = current_finger.user_id;
                connection.query('SELECT * FROM Users WHERE id = ' + current_finger.user_id, function(err, userRows, fields) {
                    if(err) throw err

                    if (userRows.length <= 0) {
                        userFinger['name'] = 'Đang cập nhật';
                        userFinger['user_code'] = 'Đang cập nhật';
                        userFinger['created_at'] = current_finger.created_at;
                        userFinger['status'] = current_finger.status;
                        fingerArr.push( userFinger );                                               
                    }
                    else { 
                        userFinger['name'] = userRows[0].name;
                        userFinger['user_code'] = userRows[0].code;
                        userFinger['created_at'] = current_finger.created_at;
                        userFinger['status'] = current_finger.status;
                        fingerArr.push( userFinger );                                                                                                                            
                    } 
                });
            }

            setTimeout(function(){ 
                res.setHeader('Content-Type', 'application/json');
                res.send(JSON.stringify({ success: true, data:fingerArr })); 
            }, 1000);                           
        }                            
    });        
});
       
router.get('/delete/(:id)', function(req, res, next) {
    var finger = { id: req.params.id }
     
    connection.query('DELETE FROM Fingers WHERE id = ' + req.params.id, finger, function(err, result) {        
        if (err) {
            req.flash('error', err)            
            res.redirect('/fingers')
        } else {


            req.flash('success', 'Finger deleted successfully! id = ' + req.params.id)            
            res.redirect('/fingers')
        }
    })
})

router.get('/get/delete/(:id)', function(req, res, next) {
    var finger = { id: req.params.id }
     
    connection.query('DELETE FROM Fingers WHERE id = ' + req.params.id, finger, function(err, result) {        
        if (err) {
            res.setHeader('Content-Type', 'application/json');
            res.send(JSON.stringify({ success: false, message: 'Delete Finger Error' })); 
        } else {
            res.setHeader('Content-Type', 'application/json');
            res.send(JSON.stringify({ success: true, message: 'Delete Finger Success' })); 
        }
    })
})
 
 
module.exports = router;