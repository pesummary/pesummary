// Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>                          
// This program is free software; you can redistribute it and/or modify it      
// under the terms of the GNU General Public License as published by the        
// Free Software Foundation; either version 3 of the License, or (at your       
// option) any later version.                                                   
//                                                                              
// This program is distributed in the hope that it will be useful, but          
// WITHOUT ANY WARRANTY; without even the implied warranty of                   
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General   
// Public License for more details.                                             
//                                                                              
// You should have received a copy of the GNU General Public License along      
// with this program; if not, write to the Free Software Foundation, Inc.,      
// 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA. 

function combines(list) {                                                        
    var loadTimer;                                                              
    var imgObject = new Image();                                                
    var heading=document.getElementsByTagName("h1")[0]                          
    var approx = heading.innerHTML.split(" ")[0]                                
    var c=document.getElementById("combines")                                     
    var ctx=c.getContext("2d")                                                  
    ctx.clearRect(0, 0, c.width, c.height);                                     
    var el= document.getElementById("corner_search").value.split(", "); 
    if ( typeof list === 'undefined' ) {                                        
        list = 'None';                                                          
    }                                                                           
    if ( list == 'None' ) {                                                     
      if ( el.length == 1 ) {                                                   
          var el = document.getElementById("corner_search").value.split(",");   
          if ( el.length == 1 ) {                                               
              var el = document.getElementById("corner_search").value.split(" ");
          }                                                                     
      }                                                                    
    } else {                                                                    
      var el = list.split(", ");                                                
    }
    c.width = 700;
    if ( el == "" ) {
        var total = document.getElementsByName("type");
        var parameters = []
        for ( var i=0; i<total.length; i++ ) {
            parameters.push(total[i].id);
        }
        var ticked = [];
        parameters = parameters.filter(function(item, pos) {
                         return parameters.indexOf(item) == pos;
                      });
        for ( var i=0; i<parameters.length; i++ ) {
            if ( document.getElementById(parameters[i]).checked == true) {
                ticked.push(parameters[i]);
            }
        }
        c.height = 520*ticked.length+50;
        for ( var i=0; i<ticked.length; i++ ) {
            if ( approx == "Comparison" ) {
                imgObject.src = '../plots/combined_posterior_'+ticked[i]+'.png'
            } else {
                imgObject.src = '../plots/1d_posterior_'+approx+'_'+ticked[i]+'.png';
            }
            ctx.drawImage(imgObject, 0, (500*i)+(i*20), 700, 500);
        }
    } else {                                                            
        c.height = 520*el.length+50;                                                                                                             
        for ( var i=0; i<el.length; i++ ) {
            if ( approx == "Comparison" ) {
                imgObject.src = '../plots/combined_posterior_'+el[i]+'.png'
            } else {                                    
                imgObject.src = '../plots/1d_posterior_'+approx+'_'+el[i]+'.png';
            }
            ctx.drawImage(imgObject, 0, (500*i)+(i*20), 700, 500);        
        }                
    }                                                                           
}
