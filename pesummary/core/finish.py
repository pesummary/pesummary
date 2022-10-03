# Licensed under an MIT style license -- see LICENSE.md

import subprocess
from pesummary.utils.utils import logger

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class FinishingTouches(object):
    """Class to handle the finishing touches

    Parameters
    ----------
    parser: argparser
        The parser containing the command line arguments
    """
    def __init__(self, inputs, **kwargs):
        self.inputs = inputs
        self.send_email()
        logger.info("Complete. Webpages can be viewed at the following url "
                    "%s" % (self.inputs.baseurl + "/home.html"))

    def send_email(self, message=None):
        """Send notification email.
        """
        if self.inputs.email:
            logger.info("Sending email to %s" % (self.inputs.email))
            try:
                self._email_notify(message)
            except Exception as e:
                logger.info("Unable to send notification email because %s" % (
                    e))

    def _email_message(self, message=None):
        """Message that will be send in the email.
        """
        if not message:
            message = ("Hi %s,\n\nYour output page is ready on %s. You can "
                       "view the result at %s \n"
                       % (self.inputs.user, self.inputs.host, self.inputs.baseurl + "/home.html"))
        return message

    def _email_notify(self, message):
        """Subprocess to send the notification email.
        """
        subject = "Output page available at %s" % (self.host)
        message = self._email_message(message)
        cmd = 'echo -e "%s" | mail -s "%s" "%s"' % (
            message, subject, self.email)
        ess = subprocess.Popen(cmd, shell=True)
        ess.wait()

    def remove_tmp_directories(self):
        """Remove the temp directories created by PESummary
        """
        from pesummary.utils import utils

        utils.remove_tmp_directories()
