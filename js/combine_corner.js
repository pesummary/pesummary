// Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
//                     Edward Fauchon-Jones <edward.fauchon-jones@ligo.org>
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3 of the License, or (at your
// option) any later version.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


function url(marker, param1, param2, approximant) {
    /* Set the url of the image object

    Parameters
    ----------
    marker: obj
        Image obj
    param1: str
        the name of one of the parameters in the density plot
    param2: str
        the name of the second parameter in the density plot
    approximant: str
        the name of the approximant that you are analysing
    */
    var ordered = [param1, param2];
    // sort the parameter names into alphabetical order
    ordered.sort();
    marker.src = '../plots/corner/'+approx+'_'+ordered[0]+'_'+ordered[1]+'_density_plot.png';
}

function draw(marker, index1, index2, ctx, length) {
    /* Place the density plot on the canvas

    Parameters
    ----------
    marker: obj
        Image object
    index1: int
        index of the column that you want the image to placed in
    index2: int
        index of the row that you want the image to be placed in
    length: int
        the number of parameters that you want to include in the corner subplot
    */
    marker.onload = function() {
        ctx.drawImage(marker, (600/length)*index1, (600/length)*index2, (600/length)-5, (600/length)-5)
    }
}

function combine() {
    /* Grab the parameters and populate a corner subplot
    */    
    // get the contents of the search bar                                         
    var el= document.getElementById("corner_search").value.split(", ");
    if ( el.length == 1 ) {
        var el = document.getElementById("corner_search").value.split(",");
        if ( el.length == 1 ) {
            var el = document.getElementById("corner_search").value.split(" ");
        }
    }
    var el=document.getElementsByTagName("h1")[0]                               
    var approx = el.innerHTML.split(" ")[0]                               
    var c=document.getElementById("canvas");                                    
    var ctx = c.getContext("2d");
    var markers = []
    // remove the previous canvas
    ctx.clearRect(0, 0, c.width, c.height);

    for (var i=0; i<el.length; i++) {
        markers[i] = new Array(el.length)
        for (var j=i; j<el.length; j++) {
            markers[i][j] = new Image();
        }
    }
    // set the source directory of the image 
    for (var i=0; i<el.length; i++) {
        for (var j=i; j<el.length; j++) {
            // if i == j then we want the histogram
            if ( i == j) {
                markers[i][j].src = '../plots/corner/'+approx+'_'+el[i]+'_histogram_plot.png'
                draw(markers[i][j], i, j, ctx, el.length)
            }
            // else we want the density plot
            else {
                url(markers[i][j], el[i], el[j], approx)
                draw(markers[i][j], i, j, ctx, el.length)
            }
        }
    }                                                                   
} 
