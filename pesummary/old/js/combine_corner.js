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


function combine(list, label="None") {
      var loadTimer;
      var imgObject = new Image();
      var header=document.getElementsByTagName("h1")[0]
      var el=document.getElementsByTagName("h7")[1]
      var approx = el.innerHTML
      var c=document.getElementById("canvas")
      var ctx=c.getContext("2d")
      c.width=700
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
      if ( el == "" ) {
          var el = [];
          var total = document.getElementsByName("type");
          var parameters = []
          for ( var i=0; i<total.length; i++ ) {
              parameters.push(total[i].id);
          }
          var ticked = [];
          for ( var i=0; i<parameters.length; i++ ) {
              if ( document.getElementById(parameters[i]).checked == true) {
                  el.push(parameters[i]);
              }
          }
      }

      function onImgLoaded(reject) {
          if (loadTimer != null) clearTimeout(loadTimer);
          if (!imgObject.complete) {
              loadTimer = setTimeout(function() {
                  onImgLoaded();
              }, 3);
          } else {
              try {
                  onPreloadComplete(c, imgObject, el);
              } catch (e) {
                  reject("Broken");
              }
          }
      }
      imgObject.src = '../plots/corner/'+label+'_'+approx+'_all_density_plots.png';
      promise = new Promise((resolve, reject) => {
          imgObject.onload = () => onImgLoaded(reject)
      })
      promise.catch((err) => window.location = "./error.html")
      c.width = 0
      c.height = 0
}

function onPreloadComplete(c, imgObject, object){
    var newImg = getImagePortion(c, imgObject, object);
  
    //place image in appropriate div
    var link = document.getElementById("mirror")
    link.src = newImg
}

function getImagePortion(c, imgObj, array){
    /* the parameters: - the image element - the new width - the new height - the x point we start taking pixels - the y point we start taking pixels - the ratio */
    // set up canvas for thumbnail
    var tnCanvas = document.createElement('canvas');
    var tnCanvasContext = tnCanvas.getContext('2d');
    tnCanvas.width = 600; tnCanvas.height = 600;
    tnCanvasContext.fillStyle = "white";
    tnCanvasContext.fillRect(0, 0, 600, 600);
   
    /* use the sourceCanvas to duplicate the entire image. This step was crucial for iOS4 and under devices. Follow the link at the end of this post to see what happens when you don’t do this */
    var bufferCanvas = document.createElement('canvas');
    var bufferContext = bufferCanvas.getContext('2d');
    bufferCanvas.width = imgObj.width;
    bufferCanvas.height = imgObj.height;
    bufferContext.drawImage(imgObj, 0, 0);

    var list = ['redshift', 'phi_jl', 'total_mass', 'chirp_mass_source', 'symmetric_mass_ratio', 'mass_1', 'ra', 'mass_2', 'tilt_2', 'mass_2_source', 'psi', 'phi_12', 'geocent_time', 'a_1', 'a_2', 'chi_p', 'phase', 'mass_1_source', 'luminosity_distance', 'chirp_mass', 'iota', 'chi_eff', 'mass_ratio', 'total_mass_source', 'tilt_1', 'dec', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'luminosity_distance', 'iota', 'dec', 'ra', 'psi', 'phase'];
    var indices = []
    
    var ratio = (157.5*3) / (array.length*210)

    for ( var i=0; i<array.length; i++) {
        if ( list.indexOf(array[i]) == -1 ) {
            alert(                                       
                "The parameter '" + array[i] + "' is not recognised. Please open the " +
                "sidebar for a full list of available parameters")
        } else {
             indices[i] = list.indexOf(array[i])
        }
    }
    indices.sort((a,b) => a-b)

    for ( var i=0; i<array.length; i++) {

        tnCanvasContext.drawImage(bufferCanvas, 100+208*indices[i]+2*(indices[i]-1), 34+208*list.length+2*(list.length-1), 208, 80, 210*i*ratio+120, 210*array.length*ratio, 208*ratio, 80*ratio)
        tnCanvasContext.drawImage(bufferCanvas, 10, 36+208*indices[i]+2*(indices[i]-1), 80, 208, 100-74*ratio, 210*i*ratio, 74*ratio, 208*ratio)
        for ( var j=i; j<array.length; j++) {
            tnCanvasContext.drawImage(bufferCanvas, 100+208*indices[i]+2*(indices[i]-1), 36+208*indices[j]+2*(indices[j]-1), 208, 208, 210*i*ratio+120, 210*j*ratio, 208*ratio, 208*ratio)
        }
    }
    return tnCanvas.toDataURL(); 
}
