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


function url(marker, param1, param2) {
    marker.src = '../plots/corner/IMRPhenomP_'+param1+'_'+param2+'_density_plot.png';
}

function draw(marker, index1, index2, ctx) {
    marker.onload = function() {
        ctx.drawImage(marker, 200*index1, 200*index2, 195, 195)
    }
}

function check_url(marker, param1, param2, ind1, ind2, ctx) {
    fetch('../plots/corner/IMRPhenomP_'+param1+'_'+param2+'_density_plot.png', {
        credentials: 'same-origin'})                                
        .then(function(response) {                                  
            if (!response.ok) {                                     
                throw Error(response.statusText);                    
            }                                                       
            return response;                                            
         })                                                          
         .then(res => {
               url(marker, param1, param2);
               draw(marker, ind1, ind2, ctx);
               })
         .catch(res => {
                url(marker, param2, param1);
                draw(marker,ind1, ind2, ctx);
                })
}

function combine() {                                                
    var el= document.getElementById("corner_search").value.split(", ");
    if ( el.length == 1 ) {
        var el = document.getElementById("corner_search").value.split(",");
        if ( el.length == 1 ) {
            var el = document.getElementById("corner_search").value.split(" ");
        }
    }
                                                   
    var c=document.getElementById("canvas");                                    
    var ctx = c.getContext("2d");
    var markers = []

    for (var i=0; i<el.length; i++) {
        markers[i] = new Array(el.length)
        for (var j=i; j<el.length; j++) {
            markers[i][j] = new Image();
        }
    }

    for (var i=0; i<el.length; i++) {
        for (var j=i; j<el.length; j++) {
            if ( i == j) {
                markers[i][j].src = '../plots/corner/IMRPhenomP_'+el[i]+'_histogram_plot.png'
                draw(markers[i][j], i, j, ctx)
            }
            else {
                check_url(markers[i][j], el[i], el[j], i, j, ctx)
            }
        }
    }                                                                   
} 
