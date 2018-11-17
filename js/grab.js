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

function _option1(approx, param) {
    window.location = "https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/html/"+approx+'_'+param+'.html'
}

function _option2(param) {
    window.location = "https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/html/"+param+'.html'
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
    fetch("https://geo2.arcca.cf.ac.uk/~c1737564/LVC/projects/bilby/GW150914/html/"+approx+"_"+param+".html", {
          credentials: 'same-origin'})
    .then(res => {
                  if ( res.status == 200 ) {
                      _option1(approx, param)
                  } else {
                      _option2(param)
                  }})
}
