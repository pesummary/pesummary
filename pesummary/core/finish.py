# Licensed under an MIT style license -- see LICENSE.md

import subprocess
import os

from pesummary.utils.utils import logger
from pesummary.core.inputs import PostProcessing

__author__ = ["Charlie Hoy <charlie.hoy@ligo.org>"]


class FinishingTouches(PostProcessing):
    """Class to handle the finishing touches

    Parameters
    ----------
    parser: argparser
        The parser containing the command line arguments
    """
    def __init__(self, inputs, **kwargs):
        super(FinishingTouches, self).__init__(inputs)
        self.send_email()
        logger.info("Complete. Webpages can be viewed at the following url "
                    "%s" % (self.baseurl + "/home.html"))

    def send_email(self, message=None):
        """Send notification email.
        """
        if self.email:
            logger.info("Sending email to %s" % (self.email))
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
                       % (self.user, self.host, self.baseurl + "/home.html"))
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
