var express = require('express');
var router = express.Router();
var connection  = require('../lib/db');
var LibLogger = require('../lib/log');
var libLogger = new LibLogger().getInstance();

router.get('/', function(req, res, next) {
     
    connection.query('SELECT * FROM Users', function(err,rows)     {
        if(err){
            req.flash('error', err); 
            return res.render('users',{page_title:"Users - Node.js",data:''});   
        }else{  
            var attendanceArr = []; 
            for (let index = 0; index < rows.length; index++) {
                const element = rows[index];
                let userAttendance = {};
                userAttendance['id'] = element.id;
                userAttendance['name'] = element.name;
                userAttendance['code'] = element.code;

                connection.query('SELECT * FROM Attendances where ( user_id = ' + element.id + ' AND DATE(created_at) = curdate() ) order by created_at limit 1', function(errInTime,inTimeRows) {
                    libLogger.log('QUERY WITH USER_ID: ' + element.id);
                    if (errInTime) {
                        req.flash('error', errInTime); 
                        return res.render('attendances',{page_title:"Attendances - Node.js",data:''});   
                    }
                                        
                    if (inTimeRows.length <= 0) {
                        libLogger.log('ADD WITH USER_ID: ' + element.id);
                        userAttendance[ 'inTime' ] = "Đang cập nhật";
                        userAttendance[ 'outTime' ] = "Đang cập nhật";
                        attendanceArr.push( userAttendance );
                    }
                    else{
                        userAttendance[ 'inTime' ] = inTimeRows[0].created_at;         
                        connection.query('SELECT * FROM Attendances where ( user_id = ' + element.id + ' AND id != ' + inTimeRows[0].id + ' AND DATE(created_at) = curdate() ) order by created_at DESC limit 1',function(errOutTime,inOutRows) {
                            if (errOutTime) {
                                req.flash('error', errOutTime); 
                                return res.render('attendances',{page_title:"Attendances - Node.js",data:''});   
                            }
                            
                            if( inOutRows.length <= 0)
                            {
                                libLogger.log('ADD WITH USER_ID: ' + element.id);
                                userAttendance[ 'outTime' ] = "Đang cập nhật";                                 
                                attendanceArr.push( userAttendance );
                            } 
                            else
                            {
                                libLogger.log('ADD WITH USER_ID: ' + element.id);
                                userAttendance[ 'outTime' ] = inOutRows[0].created_at;
                                libLogger.log('OUTTIME OF USER ID: ' + element.id + ' IS: ' + userAttendance[ 'outTime' ]);                                              
                                attendanceArr.push( userAttendance );
                            }
                        });
                    }                                       
                    libLogger.log('INTIME OF USER ID: ' + element.id + ' IS: ' + userAttendance[ 'inTime' ]);                  
                });

            }

            setTimeout(function(){ 
                
                res.render('attendances',{page_title:"Users - Node.js",data:attendanceArr}); 
            }, 1000);
        
        }                            
    });        
});

router.get('/get/list', function(req, res, next) {
     
    connection.query('SELECT * FROM Users', function(err,rows)     {
        if(err){
            res.setHeader('Content-Type', 'application/json');
            res.send(JSON.stringify({ success: false, message: 'Get Attendance List Error' }));   
        }else{  
            var attendanceArr = []; 
            for (let index = 0; index < rows.length; index++) {
                const element = rows[index];
                let userAttendance = {};
                userAttendance['id'] = element.id;
                userAttendance['name'] = element.name;
                userAttendance['code'] = element.code;

                connection.query('SELECT * FROM Attendances where ( user_id = ' + element.id + ' AND DATE(created_at) = curdate() ) order by created_at limit 1', function(errInTime,inTimeRows) {
                    libLogger.log('QUERY WITH USER_ID: ' + element.id);
                    if (errInTime) {
                        res.setHeader('Content-Type', 'application/json');
                        res.send(JSON.stringify({ success: false, message: 'Get Attendance List Error' }));   
                    }
                                        
                    if (inTimeRows.length <= 0) {
                        libLogger.log('ADD WITH USER_ID: ' + element.id);
                        userAttendance[ 'inTime' ] = "Đang cập nhật";
                        userAttendance[ 'outTime' ] = "Đang cập nhật";
                        attendanceArr.push( userAttendance );
                    }
                    else{
                        let dateObj = new Date(inTimeRows[0].created_at);
                        let dateStr = dateObj.getDate() + '/' + (dateObj.getMonth() + 1 ) + '/' + dateObj.getFullYear() + ' ' +
                            dateObj.getHours() + ':' + dateObj.getMinutes() + ':' + dateObj.getSeconds();
                        userAttendance[ 'inTime' ] = dateStr;         
                        connection.query('SELECT * FROM Attendances where ( user_id = ' + element.id + ' AND id != ' + inTimeRows[0].id + ' AND DATE(created_at) = curdate() ) order by created_at DESC limit 1',function(errOutTime,inOutRows) {
                            if (errOutTime) {
                                res.setHeader('Content-Type', 'application/json');
                                res.send(JSON.stringify({ success: false, message: 'Get Attendance List Error' })); 
                            }
                            
                            if( inOutRows.length <= 0)
                            {
                                libLogger.log('ADD WITH USER_ID: ' + element.id);
                                userAttendance[ 'outTime' ] = "Đang cập nhật";                                 
                                attendanceArr.push( userAttendance );
                            } 
                            else
                            {
                                libLogger.log('ADD WITH USER_ID: ' + element.id);

                                let dateObj = new Date(inOutRows[0].created_at);
                                let dateStr = dateObj.getDate() + '/' + (dateObj.getMonth() + 1 ) + '/' + dateObj.getFullYear() + ' ' +
                                dateObj.getHours() + ':' + dateObj.getMinutes() + ':' + dateObj.getSeconds();
                                userAttendance[ 'outTime' ] = dateStr;                                
                                libLogger.log('OUTTIME OF USER ID: ' + element.id + ' IS: ' + userAttendance[ 'outTime' ]);                                              
                                attendanceArr.push( userAttendance );
                            }
                        });
                    }                                       
                    libLogger.log('INTIME OF USER ID: ' + element.id + ' IS: ' + userAttendance[ 'inTime' ]);                  
                });

            }

            setTimeout(function(){ 
                res.setHeader('Content-Type', 'application/json');
                res.send(JSON.stringify({ success: true, data:attendanceArr }));                
            }, 1000);
        
        }                            
    });        
});
       
router.get('/delete/(:id)', function(req, res, next) {
    var finger = { id: req.params.id }
     
    connection.query('DELETE FROM Attendances WHERE id = ' + req.params.id, finger, function(err, result) {        
        if (err) {
            req.flash('error', err)            
            res.redirect('/attendances')
        } else {


            req.flash('success', 'Attendances deleted successfully! id = ' + req.params.id)            
            res.redirect('/attendances')
        }
    })
})


router.get('export/(:id)', function( req, res, next )
{
    connection.query('SELECT * FROM Users WHERE id = ' + req.params.id,function(err,rows)     {
        if(err){
            req.flash('error', err); 
            return res.render('users',{page_title:"Users - Node.js",data:''});   
        }else{  
            var attendanceArr = []; 
            for (let index = 0; index < rows.length; index++) {
                const element = rows[index];
                var userAttendance = {};
                userAttendance['id'] = element.id;
                userAttendance['name'] = element.name;
                userAttendance['code'] = element.code;

                connection.query('SELECT * FROM Attendances where ( user_id = ' + element.id + ' AND DATE(created_at) = curdate() ) order by created_at limit 1',function(errInTime,inTimeRows) {
                    
                    if (errInTime) {
                        req.flash('error', errInTime); 
                        return res.render('attendances',{page_title:"Attendances - Node.js",data:''});   
                    }
                                        
                    if (inTimeRows.length <= 0) {
                        userAttendance[ 'inTime' ] = "Đang cập nhật";
                        userAttendance[ 'outTime' ] = "Đang cập nhật";
                    }
                    else{
                        userAttendance[ 'inTime' ] = inTimeRows[0].created_at;                           
                        connection.query('SELECT * FROM Attendances where ( user_id = ' + element.id + ' AND id != ' + inTimeRows[0].id + ' AND DATE(created_at) = curdate() ) order by created_at DESC limit 1',function(errOutTime,inOutRows) {
                            if (errOutTime) {
                                req.flash('error', errOutTime); 
                                return res.render('attendances',{page_title:"Attendances - Node.js",data:''});   
                            }

                            if( inOutRows.length <= 0)
                            {
                                userAttendance[ 'outTime' ] = "Đang cập nhật";                                 
                            } 
                            else
                            {
                                userAttendance[ 'outTime' ] = inOutRows[0].created_at;
                            }                            
                        });
                    }                                       
                });

                attendanceArr.push( userAttendance );
            }

            setTimeout(function(){ 
                var json2csv = require('json2csv');
                var fields = ['User ID', 'User Name', 'Employer ID', 'In Time', 'Out Time'];
                var fieldNames = ['User ID', 'User Name', 'Employer ID', 'In Time', 'Out Time'];
                var data = json2csv({ data: docs, fields: fields, fieldNames: fieldNames });
                res.render('attendances',{page_title:"Users - Node.js",data:attendanceArr}); 
            }, 500);
        
        }                            
    });
});
 
router.post('/face-check', function( req, res, next ){    
    req.assert('image', 'Image is required').notEmpty();
    var errors = req.validationErrors();

    if( !errors ) {
        var fileName = Date.now() + '_member.png';
        var image = req.body.image;
        var base64Data = image.replace(/^data:image\/png;base64,/, "");
        console.log('========base64Data: ' + base64Data);
        require("fs").writeFile("assets/images/" + fileName, base64Data, 'base64', function(err) {
            console.log("========ERROR: " + err);

            const {PythonShell} = require('python-shell');

            var options = {
                mode: 'text',        
                pythonOptions: ['-u'],
                args: ["assets/images/" + fileName]
            };
        
            PythonShell.run('/home/huynhhq/work_space/attendance_manager/lib/face_recognization/face_recognization.py', options, function (err, result) {
                if (err) throw err;    
                    console.log('Finished');
        
                console.log(result);

                let parseStr = result[0];
                for (let index = 0; index < result[0].length; index++) {
                    parseStr = parseStr.replace('\'', '');      
                    parseStr = parseStr.replace('[', '');            
                    parseStr = parseStr.replace(']', '');  
                    parseStr = parseStr.replace(' ', '');            
                }
                
                let nameArr = parseStr.split(",");                
                if(nameArr.length <= 0)
                {
                    res.setHeader('Content-Type', 'application/json');    
                    res.send(JSON.stringify({ success: true, message: 'Not found user' }));
                }
                else
                {
                    connection.query('SELECT * FROM Users WHERE name = ' + '\'' + nameArr[0] + '\'', function(err, rows, fields) {
                        if(err) throw err
              
                        if (rows.length <= 0) {
                          libLogger.log( 'ERROR: User not found with name = ' + '\'' + nameArr[0] + '\'' );              
                        }
                        else{
                            let user_id = rows[0].id;
                            connection.query('SELECT * FROM Fingers WHERE user_id = ' + user_id, function(err, rowsFinger, fields) {
                                var currunt_time = new Date(); 
                                let finger_id = rowsFinger[0].id;
                                let user_id = rowsFinger[0].user_id;
                                var attendance_data = {
                                    user_id: user_id,
                                    finger_id: finger_id,
                                    status: 'active',
                                    created_at: currunt_time
                                };
                                connection.query('INSERT INTO Attendances SET ?', attendance_data, function(err, result) {                
                                    if (err) {
                                        libLogger.log( 'ERROR: Insert Attendances with user name = ' + '\'' + nameArr[0] + '\'' +' get err = ' + err );                                      
                                    } else {  
                                        libLogger.log( 'SUCCESS: Insert Attendances with finger = ' + '\'' + nameArr[0] + '\'' +' get err = ' + err );                                      
                                    }
                                })
                            });                                                        
                        }
                    });
    
                    res.setHeader('Content-Type', 'application/json');    
                    res.send(JSON.stringify({ success: true, message: nameArr[0] })); 
                }

            });
        });         
    }
    else {    
        res.setHeader('Content-Type', 'application/json');    
        res.send(JSON.stringify({ success: true, message:'Chua co Image then lol' }));  
    }
})

router.get('/submit-audio', function(req, res, next){    
    const {PythonShell} = require('python-shell');

    var options = {
        mode: 'text',        
        pythonOptions: ['-u']        
    };

    PythonShell.run('/home/huynhhq/work_space/attendance_manager/lib/voice_recognization/run_test_on_pi.py', options, function (err, result) {
        if (err) throw err;    
            console.log('Finished');

        console.log(result);

        let name = result[1];             
        if(name != null)       
        {
            name = name.toLowerCase();
            connection.query('SELECT * FROM Users WHERE name = ' + '\'' + name + '\'', function(err, rows, fields) {
                if(err) throw err
        
                if (rows.length <= 0) {
                    libLogger.log( 'ERROR: User not found with name = ' + '\'' + name + '\'' );              
                }
                else{
                    let user_id = rows[0].id;
                    connection.query('SELECT * FROM Fingers WHERE user_id = ' + user_id, function(err, rowsFinger, fields) {
                        var currunt_time = new Date(); 
                        let finger_id = rowsFinger[0].id;
                        let user_id = rowsFinger[0].user_id;
                        var attendance_data = {
                            user_id: user_id,
                            finger_id: finger_id,
                            status: 'active',
                            created_at: currunt_time
                        };
                        connection.query('INSERT INTO Attendances SET ?', attendance_data, function(err, result) {                
                            if (err) {
                                libLogger.log( 'ERROR: Insert Attendances with user name = ' + '\'' + name + '\'' +' get err = ' + err );                                      
                            } else {  
                                libLogger.log( 'SUCCESS: Insert Attendances with finger = ' + '\'' + name + '\'' +' get err = ' + err );                                      
                            }
                        })
                    });                                                        
                }
            });

            res.setHeader('Content-Type', 'application/json');    
            res.send(JSON.stringify({ success: true, message: name })); 
        }

    });
})

router.get('/attendance-finger/(:code)', function(req, res, next){
    let finger = req.params.code;        
    var currunt_time = new Date(); 
    connection.query('SELECT * FROM Fingers WHERE code = ' + finger, function(err, rows, fields) {
        if(err) throw err

        if (rows.length <= 0) {
            libLogger.log( 'ERROR: Fingers not found with code = ' + finger );   
            res.setHeader('Content-Type', 'application/json');    
            res.send(JSON.stringify({ success: false, message: 'failed' }));           
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
            connection.query('SELECT * FROM Users WHERE id =' + rows[0].user_id, function(err, rowsUser, fields) {
                if(err) throw err

                if (rows.length <= 0) {
                    libLogger.log( 'ERROR: User not found with code = ' + finger );   
                    res.setHeader('Content-Type', 'application/json');    
                    res.send(JSON.stringify({ success: false, message: 'failed' }));           
                }
                else
                {
                    connection.query('INSERT INTO Attendances SET ?', attendance_data, function(err, result) {                
                        if (err) {
                            libLogger.log( 'ERROR: Insert Attendances with finger = ' + finger +' get err = ' + err );                                      
                        } else {  
                            libLogger.log( 'SUCCESS: Insert Attendances with finger = ' + finger +' get err = ' + err );                                      
                            res.setHeader('Content-Type', 'application/json');    
                            res.send(JSON.stringify({ success: true, message: rowsUser[0].name }));
                        }
                    })
                }
            });                
        }
    });
});

 
module.exports = router;