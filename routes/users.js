var express = require('express');
var router = express.Router();
var connection  = require('../lib/db');
var mqttHandler = require('../lib/mqtt');
var mqttClient = new mqttHandler().getInstance();
var LibLogger = require('../lib/log');
var libLogger = new LibLogger().getInstance();

router.get('/', function(req, res, next) {

    const {PythonShell} = require('python-shell');

    var options = {
        mode: 'text',        
        pythonOptions: ['-u'],
        args: ['/home/huynhhq/work_space/attendance_manager/lib/face_recognization/temp.jpg']
    };

    PythonShell.run('/home/huynhhq/work_space/attendance_manager/lib/face_recognization/face_recognization.py', options, function (err, result) {
        if (err) throw err;    
            console.log('Finished');
            console.log('RESULTL:  ',result[1]);
        let parseStr = result[0];
        for (let index = 0; index < result[0].length; index++) {
            parseStr = parseStr.replace('\'', '');      
            parseStr = parseStr.replace('[', '');            
            parseStr = parseStr.replace(']', '');  
            parseStr = parseStr.replace(' ', '');            
        }
        
        nameArr = parseStr.split(",");
        for (let i = 0; i < nameArr.length; i++) {
            const element = nameArr[i];
            console.log(nameArr);
            
        }        
    });

    connection.query('SELECT * FROM Users ORDER BY id desc',function(err,rows)     {
        if(err){
            req.flash('error', err); 
            res.render('users',{page_title:"Users - Node.js",data:''});   
        }else{            
            res.render('users',{page_title:"Users - Node.js",data:rows});
        }                            
    });        
});

router.get('/get/list', function(req, res, next) {

    connection.query('SELECT * FROM Users ORDER BY id desc',function(err,rows)     {
        if(err){
            res.setHeader('Content-Type', 'application/json');
            res.send(JSON.stringify({ success: false, message: 'Get User List Error' })); 
        }else{            
            res.setHeader('Content-Type', 'application/json');
            res.send(JSON.stringify({ success: true, data: rows })); 
        }                            
    });        
});
 
router.get('/add', function(req, res, next){        
    res.render('users/add', {
        title: 'Add New Users',
        name: '',
        employerId: ''        
    })
})
 
router.post('/add', function(req, res, next){    
    req.assert('name', 'Name is required').notEmpty();
    req.assert('code', 'Employer is required').notEmpty(); 
  
    var errors = req.validationErrors();
     
    if( !errors ) {
        var currunt_time = new Date(); 
        var user = {
            name: req.sanitize('name').escape().trim(),
            code: req.sanitize('code').escape().trim(),
            status: 'active',
            created_at: currunt_time
        }
         
        connection.query('INSERT INTO Users SET ?', user, function(err, result) {                
            if (err) {
                req.flash('error', err);

                res.render('users/add', {
                    title: 'Add New User',
                    name: user.name,
                    code: user.code                    
                });               
            } else {  
                // Send message to IOT device
                mqttClient.sendMessage('command', '1');
                mqttClient.sendMessage('name', user.name);
                mqttClient.sendMessage('code', user.code);
                // End send message to IOT device              
                req.flash('success', 'Data added successfully!');
                res.redirect('/users');
            }
        });
    }
    else {
        var error_msg = ''
        errors.forEach(function(error) {
            error_msg += error.msg + '<br>'
        })                
        req.flash('error', error_msg)            
        res.render('users/add', { 
            title: 'Add New User',
            name: req.body.name,
            code: req.body.code
        })
    }
})
 
router.get('/edit/(:id)', function(req, res, next){
   
    connection.query('SELECT * FROM Users WHERE id = ' + req.params.id, function(err, rows, fields) {
        if(err) throw err

        if (rows.length <= 0) {
            req.flash('error', 'Users not found with id = ' + req.params.id)
            res.redirect('/users')
        }
        else {                 
            res.render('users/edit', {
                title: 'Edit user',                     
                id: rows[0].id,
                name: rows[0].name,
                code: rows[0].code                    
            })
        }            
    })
})


router.get('/add/(:name)/(:code)', function(req, res, next){
    var currunt_time = new Date(); 
    var user = {
        name: req.params.name,
        code: req.params.code,
        status: 'active',
        created_at: currunt_time
    }
    connection.query('SELECT * FROM Users WHERE code = ' + req.params.code, function(err, rows, fields) {
        if (rows.length > 0) {
            res.setHeader('Content-Type', 'application/json');
            res.send(JSON.stringify({ success: false, message: 'Duplicated User' }));    
        }
        else
        {
            connection.query('INSERT INTO Users SET ?', user, function(err, result) {                
                if (err) {
                    res.setHeader('Content-Type', 'application/json');
                    res.send(JSON.stringify({ success: false, message: 'Add User Error' }));               
                } else {  
                    // Send message to IOT device
                    mqttClient.sendMessage('command', '1');
                    mqttClient.sendMessage('name', user.name);
                    mqttClient.sendMessage('code', user.code);
                    // End send message to IOT device  
                    res.setHeader('Content-Type', 'application/json');
                    res.send(JSON.stringify({ success: true, message: 'Add User Successfull' }));                           
                }
            });
        }
    });    

});
 
router.post('/update/:id', function(req, res, next) {
    req.assert('name', 'Name is required').notEmpty();
    req.assert('code', 'Name is required').notEmpty();    
  
    var errors = req.validationErrors()
     
    if( !errors ) {   
 
        var user = {
            name: req.sanitize('name').escape().trim(),
            code: req.sanitize('code').escape().trim()
        }
         
        connection.query('UPDATE Users SET ? WHERE id = ' + req.params.id, user, function(err, result) {            
            if (err) {
                req.flash('error', err)
                                    
                res.render('Users/edit', {
                    title: 'Edit User',
                    id: req.params.id,
                    name: req.body.name,
                    code: req.body.code
                })
            } else {
                req.flash('success', 'Data updated successfully!');
                res.redirect('/users');
            }
        });         
    }
    else {
        var error_msg = ''
        errors.forEach(function(error) {
            error_msg += error.msg + '<br>'
        })
        req.flash('error', error_msg)         
        res.render('users/edit', { 
            title: 'Edit User',            
            id: req.params.id, 
            name: req.body.name,
            code: req.body.code
        })
    }
})
       
router.get('/delete/(:id)', function(req, res, next) {
    var user = { id: req.params.id }
     
    connection.query('DELETE FROM Users WHERE id = ' + req.params.id, user, function(err, result) {        
        if (err) {
            req.flash('error', err)            
            res.redirect('/users')
        } else {

            connection.query('SELECT * FROM Fingers WHERE user_id = ' + req.params.id, function(err, fingerRows, fields) {
                if(err) throw err

                if (fingerRows.length <= 0) {
                    libLogger.log( 'ERROR: Fingers not found with user id = ' + req.params.id  );              
                }
                else{
                    libLogger.log( 'SUCCESS: Delete Fingers with user id = ' + req.params.id  );              
                    mqttClient.sendMessage('command', '2');
                    mqttClient.sendMessage('deletecode', fingerRows[0].code);

                    connection.query('DELETE FROM Fingers WHERE id = ' + fingerRows[0].id);
                }
            });
            req.flash('success', 'User deleted successfully! id = ' + req.params.id)            
            res.redirect('/users')
        }
    })
})

router.get('/get/delete/(:id)', function(req, res, next) {
    var user = { id: req.params.id }
     
    connection.query('DELETE FROM Users WHERE id = ' + req.params.id, user, function(err, result) {        
        if (err) {
            res.setHeader('Content-Type', 'application/json');
            res.send(JSON.stringify({ success: false, message: 'Delete User Error' })); 
        } else {

            connection.query('SELECT * FROM Fingers WHERE user_id = ' + req.params.id, function(err, fingerRows, fields) {
                if(err) throw err

                if (fingerRows.length <= 0) {
                    libLogger.log( 'ERROR: Fingers not found with user id = ' + req.params.id  );              
                }
                else{
                    libLogger.log( 'SUCCESS: Delete Fingers with user id = ' + req.params.id  );              
                    mqttClient.sendMessage('command', '2');
                    mqttClient.sendMessage('deletecode', fingerRows[0].code);

                    connection.query('DELETE FROM Fingers WHERE id = ' + fingerRows[0].id);
                }
            });
            res.setHeader('Content-Type', 'application/json');
            res.send(JSON.stringify({ success: true, message: 'Delete Finger Success' })); 
        }
    })
})
 
 
module.exports = router;