// Copyright (C) 2018  Charlie Hoy <charlie.hoy@ligo.org>
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

function _option1(approx, param, home) {
    /* Open a webpage that is of the form "./html/approx_param.html"

    Parameters
    ----------
    approx: str
        name of the approximant that you are analysing
    param: str
        name of the parameter that you wish to analyse
    home: bool
        if True, we are on the home page
    */
    if ( home == "True" ) {
        window.location = "./html/"+approx+'_'+param+'.html'
    } else {
        window.location = "../html/"+approx+'_'+param+'.html'
    }
}

function _option2(param, home) {
    /* Open a webpage that is of the form "./html/param.html"

    Parameters
    ----------
    param: str
        name of the parameter that you wish to analyse
    home: bool
        if True, we are on the home page
    */
    if ( home == "True" ) {
        window.location = "./html/"+param+'.html'
    } else {
        window.location = "../html/"+param+'.html'
    }
}

function grab_html(param) {
    /* Return the url of a parameters home page for a given approximant

    Parameters
    ----------
    param: str
        name of the parameter that you want to link to
    */
    var el=document.getElementsByTagName("h1")[0]
    var approx = el.innerHTML.split(" ")[0]
    // See if we are on the home page or not.
    if ( el.innerHTML.split(" ")[0] == "Parameter" ) {
        var link = "./html/"+approx+"_"+param+".html"
        var home = "True"
    } else {
        var link = "../html/"+approx+"_"+param+".html"
        var home = "False"
    }
    // There are two formats for html names and two options for where they
    // are located depending on your tab. Here we check to see if a webpage of 
    // one format exists, and if it does (res.status = 200) then we open that
    // webpage. Otherwise we know that it is the other form and open that
    fetch(link, {credentials: 'same-origin'})
    .then(res => {
                  if ( res.status == 200 ) {
                      _option1(approx, param, home)
                  } else {
                      _option2(param, home)
                  }})
}
