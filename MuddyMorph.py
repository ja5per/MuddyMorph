#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MuddyMorph Wizard App.

A Fuzzy Inbetween Generator For Lazy Animators.

This module holds the Qt application, a modest graphical user interface.
The wizard guides the user through all steps required for creating inbetween
frames for a juicy and fluid morph animation.

The current implementation embodies a bit of a compromise between
user friendliness on one hand (intuitive, eloquent, robust, etcetera bla bla),
and easy programming on the other hand (low complexity, and a bit sloppy too).
"""

# Dependencies: Open source
from os import path
import sys, traceback
from time import time, asctime
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow,
                             QMessageBox , QFileDialog, QColorDialog)

# Dependencies: Home grown
import muddymorph_go as gogo
import muddymorph_algo as algo
from muddymorph_ui import Ui_MainWindow

# Project meta info
__author__   = algo.__author__
__version__  = algo.__version__
__revision__ = algo.__revision__


class Gui(QMainWindow):
    """
    MuddyMorph User Interface.
    """

    def __init__(self, app, parent=None):
        """
        Initialize User Interface:

            1. Set up the window.
            2. Load or initialize user settings.
            3. Define connections.
            4. Hide all but the first tab.
            5. Say hello through the status bar.
        """
        print(gogo.timestamp() + 'Setting up user interface')

        # Get that user interface going
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Reserve a cosy corner for computational stuff
        self.blankSlate()

        # Accept dragging-n-dropping of image files into the window
        self.setAcceptDrops(True)

        # Set up a frame rate clock for movie playback
        self.movieTimer = QtCore.QBasicTimer()
        self.stopMotion()

        # Fetch user preferences from previous session
        self.settingsfile = path.join(gogo.temppath(), '..', 'muddymorph.json')
        if path.exists(self.settingsfile):
            self.settings = gogo.load_settings(self.settingsfile)
        else:
            self.settings = gogo.default_settings()

        print(gogo.timestamp() + 'Temporary file path is ' + \
              self.settings['temppath'])
        if not path.exists(self.settings['temppath']):
            gogo.init_temp(self.settings)

        # Remember the key frame directory from last time
        keys = self.settings['keyframes']
        self.keyFolder = path.expanduser('~')
        if len(keys) > 0:
            if path.isfile(keys[0]):
                self.keyFolder = path.dirname(keys[0])

        # Continue with previous project, or start afresh?
        s = self.settings['render']
        if len(keys) > 0 and self.settings['step'] > 1:
            if s['name'] in ['morph', 'muddymorph']:
                name = 'your previous project'
            else:
                name = s['name']

            q = 'Would you like to continue with {}?'.format(name)
            answer = self.askYesNo(question=q, cancancel=False, defans='no')

            if answer != 'yes':
                msg    = 'Starting a new project '
                msg   += '(with preferences from last session {})'
                msg    = msg.format(s['name'])
                print(gogo.timestamp() + msg)

                self.settings['step'] = 1
                self.settings['keyframes'] = []

            else:
                msg = 'Continuing with current project '
                print(gogo.timestamp() + msg + s['name'])

        # Prepare the GUI for action
        self.getItOn()


    def getItOn(self, event=None, doconnect=True):
        """
        Apply project settings. These can be:
            - Default settings,
            - Last session settings,
            - Or settings dropped from any old project file.
        """
        s = self.settings['render']
        haskeys = len(self.settings['keyframes']) > 0

        # Apply global preferences for first step
        self.ui.radioLoop.setChecked(s['loop'])

        # Make sure there is at least one button that can be clicked
        self.enableButtons()
        self.ui.buttonFinishKey.setEnabled(haskeys)

        # Show logo in preview label widgets
        self.showBitmap(self.ui.previewKey    )
        self.showBitmap(self.ui.previewPrefs  )
        self.showBitmap(self.ui.previewEdge   )
        self.showBitmap(self.ui.previewTraject)
        self.showBitmap(self.ui.previewMotion )
        self.showBitmap(self.ui.previewRender )

        # Make sure the bitmaps resize whenever the whole window does
        self.ui.previewKey    .resizeEvent = self.resizeAllBitmaps
        self.ui.previewEdge   .resizeEvent = self.resizeAllBitmaps
        self.ui.previewPrefs  .resizeEvent = self.resizeAllBitmaps
        self.ui.previewMotion .resizeEvent = self.resizeAllBitmaps
        self.ui.previewRender .resizeEvent = self.resizeAllBitmaps
        self.ui.previewTraject.resizeEvent = self.resizeAllBitmaps

        # Empty key frame and sequence selection lists
        self.fillTables()

        # Set up connections for all tabs
        if doconnect:
            self.ui.buttonSelectFiles.clicked.connect(self.selectKeys  )
            self.ui.radioLoop        .clicked.connect(self.switchLoop  )
            self.ui.buttonFinishKey  .clicked.connect(self.finishKey   )
            self.ui.buttonFolder     .clicked.connect(self.selectFolder)
            self.ui.buttonRender     .clicked.connect(self.startRender )
            self.ui.buttonAbortRender.clicked.connect(self.abortRender )

            self.connectPrefs  ()
            self.connectEdge   ()
            self.connectTraject()
            self.connectMotion ()

        # Show only the tabs that need to be shown
        self.showTabs()

        # Put toggles and switches in the right positions
        tab = self.settings['step']
        if tab > 1:
            self.ui.tabPreferences.setFocus()
            self.finishKey(moveon=False)
        if tab > 2:
            self.ui.tabEdge.setFocus()
            self.finishPrefs(moveon=False)
        if tab > 3:
            self.ui.tabTrajectories.setFocus()
            self.finishEdge(moveon=False)

        # Give a hint to the user as what to do next
        if tab == 1:
            msg = 'Greetings! Please select a series of key frame image files.'
        else:
            msg = 'Let us continue where we left off last time.'
        self.ui.statusBar.showMessage(msg)


    def blankSlate(self):
        """
        Clear data containers.
        """
        self.gridX  = None # Mesh grid x-coordinates
        self.gridY  = None # Mesh grid y-coordinates
        self.K      = None # Representative key bitmap
        self.G      = None # Background bitmap (backdrop + color)
        self.B      = None # Backdrop bitmap (possibly transparent)
        self.comput = None # Computational thread (one at a time)


    def busyBusy(self):
        """
        Check whether the worker thread is currently busy or not.
        """
        if self.comput is None:
            return False
        else:
            return self.comput.isRunning()


    def reportError(self, exception, trace):
        """
        Some error occurred mid-wizard.
        Display the error message and traceback info,
        and write it to *muddymorph_lasterror.log*
        """

        # Whatever happened, it is over now ...
        self.enableButtons()

        # Convert exception and trace info to presentable strings
        flavour   = str(exception[0])
        exception = str(exception[1])
        trace     = str(trace)
        flavour   = flavour.replace("<class '", "").replace("'>", "").strip()

        # Show exception info in terminal
        print(trace)
        print('')

        # Close lingering thread
        # No need for politeness or diplomacy here;
        # just terminate the bloody crash causing basterd.
        if not self.comput is None and not \
            self.comput.isFinished():
            self.comput.terminate()
        self.comput = None

        # Write report to file
        with open('muddymorph_lasterror.log', 'w') as doc:
            doc.write('MuddyMorph Error Report\n')
            doc.write(asctime() + '\n\n')
            doc.write(flavour   + '\n\n')
            doc.write(exception + '\n\n')
            doc.write(trace     + '\n\n')

        # Pop up in your face!
        hdr  = 'MuddyMorph - Error! Kaboom!'
        msg  = 'Something seems to have gone terribly wrong.\n'
        msg += '{}: {}'.format(flavour, exception)
        QMessageBox.critical(self, hdr, msg)


    def dragEnterEvent(self, event):
        """
        Enable dragging & dropping files into the application.
        """
        if event.mimeData().hasUrls:
            event.accept()
        else:
            print(gogo.timestamp() + "Ignoring drop without URL")
            event.ignore()


    def dropEvent(self, event):
        """
        Dropped files become the new key frames.
        """
        filelist = []
        for url in event.mimeData().urls():
            f = str(url.toLocalFile())
            if path.isfile(f):
                filelist.append(f)

        msg = '{} files have been dropped'
        print(gogo.timestamp() + msg.format(len(filelist)))

        if len(filelist) == 1 and filelist[0].lower().endswith('.json'):
            self.settings = gogo.load_settings(filelist[0])
            msg = 'Continuing with ' + self.settings['render']['name']
            self.ui.statusBar.showMessage(msg)
            print(gogo.timestamp() + msg)
            gogo.init_temp(self.settings)
            self.getItOn(doconnect=False)

        elif len(filelist) > 0:
            self.updateKeys(filelist)


    def enableButtons(self, enabled=True):
        """
        Enable or disable all user controls.
        During computationally intensive steps (threads) controls are best
        temporarily disabled to prevent impatient users clicking like madmen.

        Note that most preview abort buttons are antagonistic;
        they are enabled when the rest is disabled, and vice versa.
        """
        haskeys  = len(self.settings['keyframes']) > 0
        hasmovie = len(self.movieReel) > 0
        hasblob  = self.ui.checkBlob.isChecked()

        if enabled: previewtext = u'Preview  ☻'
        else      : previewtext = u'Abort  ■'

        # Step 1. Images
        self.ui.buttonSelectFiles  .setEnabled(enabled)
        self.ui.tableKey           .setEnabled(enabled)
        self.ui.radioStraight      .setEnabled(enabled)
        self.ui.radioLoop          .setEnabled(enabled)
        self.ui.radioReverse       .setEnabled(enabled)
        self.ui.buttonFinishKey    .setEnabled(enabled and haskeys)

        # Step 2. Settings
        self.ui.comboExtension     .setEnabled(enabled)
        self.ui.checkTransparency  .setEnabled(enabled)
        self.ui.spinQuality        .setEnabled(enabled)
        self.ui.spinTypicalTweens  .setEnabled(enabled)
        self.ui.buttonBackground   .setEnabled(enabled)
        self.ui.checkAutoBack      .setEnabled(enabled)
        self.ui.buttonLineColor    .setEnabled(enabled)
        self.ui.checkLineColor     .setEnabled(enabled)

        # Step 3. Edges
        self.ui.tableEdge          .setEnabled(enabled)
        self.ui.comboChannel       .setEnabled(enabled)
        self.ui.spinBlur           .setEnabled(enabled)
        self.ui.spinThreshold      .setEnabled(enabled)
        self.ui.checkScharr        .setEnabled(enabled)
        self.ui.checkInvert        .setEnabled(enabled)
        self.ui.checkShowSilhouette.setEnabled(enabled)
        self.ui.checkShowEdge      .setEnabled(enabled)
        self.ui.checkShowCenter    .setEnabled(enabled)
        self.ui.buttonEdge         .setEnabled(enabled)
        self.ui.buttonAllEdge      .setEnabled(enabled)
        self.ui.buttonFinishEdge   .setEnabled(enabled)

        # Step 4. Trajectories
        self.ui.tableTraject       .setEnabled(enabled)
        self.ui.spinMaxPoints      .setEnabled(enabled)
        self.ui.spinNeighbours     .setEnabled(enabled)
        self.ui.checkCorners       .setEnabled(enabled)
        self.ui.comboCornerCatcher .setEnabled(enabled)
        self.ui.checkArc           .setEnabled(enabled)
        self.ui.checkSpin          .setEnabled(enabled)
        self.ui.checkSilhouette    .setEnabled(enabled)
        self.ui.spinDetail         .setEnabled(enabled)
        self.ui.spinSimiLim        .setEnabled(enabled)
        self.ui.spinMaxMove        .setEnabled(enabled)
        self.ui.buttonAllTraject   .setEnabled(enabled)
        self.ui.buttonFinishTraject.setEnabled(enabled)
        self.ui.buttonTraject      .setText(previewtext)

        # Step 5. Motion
        self.ui.tableMotion        .setEnabled(enabled)
        self.ui.buttonRevPrev      .setEnabled(enabled)
        self.ui.spinVignette       .setEnabled(enabled)
        self.ui.checkBlob          .setEnabled(enabled)
        self.ui.spinHardness       .setEnabled(enabled and hasblob)
        self.ui.spinInflate        .setEnabled(enabled and hasblob)
        self.ui.spinBetweens       .setEnabled(enabled)
        self.ui.comboMotion        .setEnabled(enabled)
        self.ui.spinFade           .setEnabled(enabled)
        self.ui.buttonAllMotion    .setEnabled(enabled)
        self.ui.buttonFinishMotion .setEnabled(enabled)
        self.ui.spinFrameRate      .setEnabled(enabled and hasmovie)
        self.ui.buttonPlayMotion   .setEnabled(enabled and hasmovie)
        self.ui.slideMotion        .setEnabled(enabled and hasmovie)
        self.ui.buttonMotion       .setText(previewtext)

        # Step 6. Render
        self.ui.editFolder         .setEnabled(enabled)
        self.ui.buttonFolder       .setEnabled(enabled)
        self.ui.editName           .setEnabled(enabled)
        self.ui.buttonRender       .setEnabled(enabled)
        self.ui.buttonAbortRender  .setEnabled(not enabled)


    def askYesNo(self, question='Are you bloody sure?',
                 caption='MuddyMorph - Question',
                 cancancel=True, defans='yes'):
        """
        Ask a yes-or-no question through a popup dialog.
        Returns a short string; 'yes', 'no', or 'cancel'.
        """
        yes    = QMessageBox.Yes
        no     = QMessageBox.No
        cancel = QMessageBox.Cancel

        if defans.lower() == 'yes': da = yes
        elif cancancel and defans.lower() == 'cancel': da = cancel
        else: da = no

        options = yes | no | cancel if cancancel else yes | no

        response = QMessageBox.question(self, caption, question, options, da)

        if   response == yes: return 'yes'
        elif response == no : return 'no'
        else                : return 'cancel'


    def showTabs(self, step = None):
        """
        Show or hide tabs.

        New tabs appear upon pressing the various 'next' buttons.
        Tabs are hidden again when the user goes back and changes something.
        """

        # In case of any tab change stop and disable movie playback
        self.stopMotion()

        # Do we need to do unveil or hide anything?
        count = self.ui.tabWidget.count()
        if step is None:
            step = self.settings['step']
        else:
            self.settings['step'] = step
        if step < 1:
            return
        elif count == step:
            self.ui.tabWidget.setCurrentIndex(step - 1)
            return

        # Show what's going on
        print(gogo.timestamp() + 'Revealing tabs for step <= {}'.format(step))

        # Save settings to file
        gogo.save_settings(self.settings, self.settingsfile)

        # Hide tabs that need to be hidden
        # And scrap corresponding temporary files
        for hide in range(count - 1, step - 1, -1):
            self.ui.tabWidget.removeTab(hide)

        # Reveal tabs that need to be revealed
        if step > count:
            for t, w, s in [(2, self.ui.tabPreferences , 'Preferences' ),
                            (3, self.ui.tabEdge        , 'Edge'        ),
                            (4, self.ui.tabTrajectories, 'Trajectories'),
                            (5, self.ui.tabMotion      , 'Motion'      ),
                            (6, self.ui.tabRender      , 'Render'      )]:

                if step >= t and count < t:
                    icon = QtGui.QIcon(':stilo/guicandy/t{}.png'.format(t))
                    self.ui.tabWidget.insertTab(t - 1, w, icon, s)

        # Set focus to the last tab (current step)
        self.ui.tabWidget.setCurrentIndex(step - 1)


    def showBitmap(self, widget, image=None):
        """
        Load a bitmap from file and show it
        in one of the various preview panes (label widgets).
        Supply the widget object and the image filename.
        """
        if image is None or image == '' or image == 'logo':
            image   = ':stilo/guicandy/logo.png'
        elif image == 'wait':
            image   = ':stilo/guicandy/wait.png'
        elif image == 'icon':
            image   = ':stilo/guicandy/icon.png'
        elif not path.exists(image):
            image   = ':stilo/guicandy/missing.png'

        bitmap = QtGui.QPixmap(image)
        widget.pic = bitmap
        widget.setPixmap(bitmap)
        self.resizeBitmap(widget)


    def resizeBitmap(self, widget=None):
        """
        Resize a bitmap display widget while preserving the aspect ratio.
        """
        if not widget: return
        try:
            scaled = widget.pic.scaled(widget.size(),
                                       QtCore.Qt.KeepAspectRatio,
                                       QtCore.Qt.SmoothTransformation)
            widget.setPixmap(scaled)
        except AttributeError:
            pass


    def resizeAllBitmaps(self, event=None):
        """
        Redraw all visible preview panes when the window changes shape.
        """
        n = self.ui.tabWidget.count()

        if n >= 1: self.resizeBitmap(self.ui.previewKey    )
        if n >= 2: self.resizeBitmap(self.ui.previewPrefs  )
        if n >= 3: self.resizeBitmap(self.ui.previewEdge   )
        if n >= 4: self.resizeBitmap(self.ui.previewTraject)
        if n >= 5: self.resizeBitmap(self.ui.previewMotion )
        if n >= 6: self.resizeBitmap(self.ui.previewRender )


    def setupTable(self, widget, tableheader, tabledata):
        """
        Put content into a table widget.
        """
        m = TableModel(tabledata, tableheader, self)
        widget.setModel(m)
        widget.resizeColumnsToContents()
        widget.horizontalHeader().setStretchLastSection(True)
        widget.setMinimumHeight(100)
        widget.selectRow(0)


    def fillTables(self, dokey=True, doseq=True, doconnect=True):
        """
        (Re)populate all key and sequence selection tables.
        """
        keycount  = len(self.settings['keyframes'])
        hdr_key   = ['#', 'Key frame' ]
        hdr_morph = ['#', 'From', 'To']

        # No key frames? Empty all tables then
        if keycount == 0:
            for widget in [self.ui.tableKey ,
                           self.ui.tableEdge]:
                self.setupTable(widget, hdr_key, [[''] * 2])
            for widget in [self.ui.tableTraject,
                           self.ui.tableMotion ]:
                self.setupTable(widget, hdr_morph, [[''] * 3])
            return

        # Key selection table content
        keydata = []
        for i in range(keycount):
            name = path.basename(self.settings['keyframes'][i])
            name, _ = path.splitext(name)
            keydata.append([i + 1, name])

        # Fire up key selection tables
        if dokey:
            for widget, slot in [(self.ui.tableKey , self.switchKey ),
                                 (self.ui.tableEdge, self.switchEdge)]:
                self.setupTable(widget, hdr_key, keydata)
                model = widget.selectionModel()
                if doconnect: model.selectionChanged.connect(slot)
                else        : model.selectionChanged.disconnect(slot)

        if keycount == 1: return

        # Fill and fire up sequence tables
        if doseq:
            seqdata = []
            for i in range(keycount - 1):
                seqdata.append([i + 1, keydata[i][-1], keydata[i + 1][-1]])
            if self.settings['render']['loop']:
                i += 1
                seqdata.append([i + 1, keydata[-1][-1], keydata[0][-1]])
            for widget, slot in [(self.ui.tableTraject, self.switchTraject),
                                 (self.ui.tableMotion , self.switchMotion )]:
                self.setupTable(widget, hdr_morph, seqdata)
                model = widget.selectionModel()
                if doconnect: model.selectionChanged.connect(slot)
                else        : model.selectionChanged.disconnect(slot)


    def selectKeys(self):
        """
        Select key frame image files through a dialog window.
        """

        # Select files
        print(gogo.timestamp() + 'Select key frames ... ', end='')
        filez, _ = QFileDialog.getOpenFileNames(self, \
                   directory = self.keyFolder, \
                   caption   = 'MuddyMorph - Select key frames', \
                   filter    = 'Images (*.png *.jpg)\nAll files (*.*)')

        # Convert to a simple list of filenames (instead of misty object)
        filelist = [str(f) for f in filez]
        if len(filelist) == 0:
            print('no files selected')
            return
        print('{} image files selected'.format(len(filelist)))

        self.updateKeys(filelist)


    def updateKeys(self, filelist):
        """
        A fresh list of files has arrived through dialog or dragging-n-dropping.
        Perform a quick image size consistency check, and proceed if okay.
        """

        # New keys means a fresh start
        # Forget the old shit, let the new shit begin
        self.blankSlate()

        # There may be more where that came from
        self.keyFolder = path.dirname(filelist[0])

        # All images must have the same dimensions
        try:
            ok = gogo.size_consistency_check(filelist)
        except:
            self.reportError(sys.exc_info(), traceback.format_exc())
        if not ok:
            hdr  = 'MuddyMorph - Image mischief'
            msg  = 'Inconsistent image dimensions.\n'
            msg += 'All key frames must be of the same size.'
            QMessageBox.critical(self, hdr, msg)
            return

        # Store selection
        self.settings['keyframes'] = filelist

        # Hide all but the first tab
        self.showTabs(1)

        # Populate selection tables
        self.fillTables()

        # Show the frame that just so happens to be selected
        self.switchKey()

        # Enable next button if we have at least two key frames
        if len(filelist) < 2:
            msg = 'At least two key frames are required'
            self.ui.statusBar.showMessage(msg)
        else:
            msg  = 'Now please select a representative key frame '
            msg += 'as a default for previewing.'
            self.ui.statusBar      .showMessage(msg)
            self.ui.radioLoop      .setEnabled(True)
            self.ui.buttonFinishKey.setEnabled(True)


    def switchKey(self):
        """
        A new main key frame has just been selected.
        Better show it in the preview pane.
        """
        k     = self.ui.tableKey.currentIndex().row()
        image = self.settings['keyframes'][k]
        msg   = 'Key frame {} has been chosen as representative'
        print(gogo.timestamp() + msg.format(k + 1))
        self.showBitmap(self.ui.previewKey, image)


    def switchLoop(self):
        """
        Enable or disable loop.

        In case of a looped animation an additional morph sequence
        - from the last to the first key frame - will be appended.
        """

        if len(self.settings['keyframes']) == 0: return

        isloop = self.ui.radioLoop.isChecked()

        if isloop:
            print(gogo.timestamp() + "Loop enabled")
        else:
            print(gogo.timestamp() + "Loop disabled")

        self.settings['render']['loop'] = isloop
        self.showTabs(1)
        self.fillTables(dokey=False, doseq=True)


    def finishKey(self, event=None, moveon=True):
        """
        Finish key frame selection (step 1).
        """

        # Create directories for storing intermediate results
        gogo.init_temp(self.settings)

        # Short hand name for render settings
        s = self.settings['render']

        # Do not fire any signals by accident in here
        self.connectPrefs(False)

        if moveon:
            print(gogo.timestamp() + 'Moving on to step 2')

            # Store loop setting
            s['loop'   ] = self.ui.radioLoop   .isChecked()
            s['reverse'] = self.ui.radioReverse.isChecked()

            # The primary key will come in handy on many occasions
            if self.K is None: self.fetchPrimeKey()

            # Suggest export file type
            _, s['ext'] = path.splitext(self.settings['keyframes'][0])
            s['ext'] = s['ext'].replace('.', '').lower()

            # Suggest transparency if the primary key
            # has an active alpha channel
            if s['ext'] == 'png' and self.K[:, :, 3].min() < 1:
                s['transparent'] = True
                s['autoback'] = False

            # Suggest project name based on common part of key file names
            keyz = [path.basename(f) for f in self.settings['keyframes']]
            s['name'] = path.commonprefix(keyz).strip('0-_ ')
            if not s['name']: s['name'] = 'morph'

            # Sample border in case of automatic background color
            if s['autoback']: s['backcolor'] = algo.bordercolor(self.K)

        # Make sure controls  match export extension
        if s['ext'] == 'png':
            self.ui.comboExtension.setCurrentIndex(0)
            self.ui.spinQuality.setEnabled(False)
            self.ui.checkTransparency.setEnabled(True)
        else:
            s['ext'], s['transparent'] = 'jpg', False
            self.ui.comboExtension.setCurrentIndex(1)
            self.ui.spinQuality.setEnabled(True)
            self.ui.checkTransparency.setEnabled(False)
            self.ui.checkTransparency.setChecked(False)

        # Apply preferences
        tweens = self.settings['motion']['inbetweens']
        tweens = algo.most_frequent_value(tweens)
        self.ui.spinTypicalTweens.setValue(tweens)
        self.ui.editName         .setText   (s['name'       ])
        self.ui.checkTransparency.setChecked(s['transparent'])
        self.ui.checkAutoBack    .setChecked(s['autoback'   ])
        self.ui.checkLineColor   .setChecked(s['lineart'    ])
        self.ui.spinQuality      .setValue  (s['quality'    ])
        self.setBG(self.ui.buttonBackground, s['backcolor'  ])
        self.setBG(self.ui.buttonLineColor , s['linecolor'  ])
        if s['backdrop']: self.ui.buttonBackdrop.setText(s['backdrop'])

        self.showBackdrop()
        self.connectPrefs(True)

        # Move on to the next step
        if not moveon: return
        msg = 'This is a good moment to pick project properties'
        self.showTabs(2)
        self.ui.statusBar.showMessage(msg)


    def fetchPrimeKey(self):
        """
        Load representative key bitmap, and set up mesh grid.
        Keeping these things in memory saves a bit of precious time.
        """
        print(gogo.timestamp() + 'Loading representative key')
        k = self.ui.tableKey.currentIndex().row()
        self.K = algo.load_rgba(self.settings['keyframes'][k])
        self.gridX, self.gridY = algo.grid(self.K)


    def connectPrefs(self, connect=True):
        """
        Make or break all connections in the preferences tab,
        with the exception of the proceed button.
        Temporarily disabling signals is needed when applying preferences.
        """
        if connect:
            self.ui.comboExtension.currentIndexChanged.connect(self.refreshPrefs)
            self.ui.spinQuality      .valueChanged.connect(self.refreshPrefs)
            self.ui.spinTypicalTweens.valueChanged.connect(self.refreshPrefs)
            self.ui.checkTransparency.clicked.connect(self.refreshPrefs  )
            self.ui.buttonBackground .clicked.connect(self.pickBackground)
            self.ui.checkAutoBack    .clicked.connect(self.autoBackground)
            self.ui.buttonLineColor  .clicked.connect(self.pickLineColor )
            self.ui.buttonBackdrop   .clicked.connect(self.selectBackdrop)
            self.ui.buttonFinishPrefs.clicked.connect(self.finishPrefs   )
        else:
            self.ui.comboExtension.currentIndexChanged.disconnect(self.refreshPrefs)
            self.ui.spinQuality      .valueChanged.disconnect(self.refreshPrefs)
            self.ui.spinTypicalTweens.valueChanged.disconnect(self.refreshPrefs)
            self.ui.checkTransparency.clicked.disconnect(self.refreshPrefs  )
            self.ui.buttonBackground .clicked.disconnect(self.pickBackground)
            self.ui.checkAutoBack    .clicked.disconnect(self.autoBackground)
            self.ui.buttonLineColor  .clicked.disconnect(self.pickLineColor )
            self.ui.buttonBackdrop   .clicked.disconnect(self.selectBackdrop)
            self.ui.buttonFinishPrefs.clicked.disconnect(self.finishPrefs   )


    def refreshPrefs(self, event=None, showbg=True):
        """
        Tidy up global project preferences:

        - JPG supports lossy compression (quality), but not transparency.
        - PNG supports transparency, but not lossy compression.
        - Show project duration ballpark estimate in the status bar.
        """
        self.showTabs(2)

        # Current settings
        name  = str(self.ui.editName.text())
        ext   = str(self.ui.comboExtension.currentText()).lower()
        trans = self.ui.checkTransparency.isChecked()
        q     = self.ui.spinQuality.value()
        la    = self.ui.checkLineColor.isChecked()

        # Quality is for JPG, and transparency is for PNG
        if ext == 'png':
            self.ui.spinQuality.setEnabled(False)
            self.ui.checkTransparency.setEnabled(True)
        elif ext == 'jpg':
            self.ui.spinQuality.setEnabled(True)
            self.ui.checkTransparency.setEnabled(False)
            if trans:
                trans = False
                self.ui.checkTransparency.setChecked(False)

        # Remember current settings
        self.settings['render']['name'       ] = name
        self.settings['render']['ext'        ] = ext
        self.settings['render']['transparent'] = trans
        self.settings['render']['quality'    ] = q
        self.settings['render']['lineart'    ] = la

        # Update the background preview
        if showbg:
            if not self.sender() in (self.ui.spinQuality      ,
                                     self.ui.spinFrameRate    ,
                                     self.ui.spinTypicalTweens):
                self.showBackdrop()

        # Show a rough estimate of the animation duration
        # (Can be helpful in deciding on the typical number of inbetweens)
        morphs      = gogo.count_morphs(self.settings)
        tweens      = int(self.ui.spinTypicalTweens.value())
        totalframes = morphs * (tweens + 1) + 1
        fps         = 24
        stopwatch   = 1. * totalframes / fps
        ballpark    = 'Animation sequence duration will be roughly '
        ballpark   += '{0:.0f} frames'.format(totalframes)
        ballpark   += ' or {} @ {} fps'.format(gogo.duration(stopwatch), fps)

        self.ui.statusBar.showMessage(ballpark)


    def getBG(self, widget):
        """
        Return the background color from a widget stylesheet.
        Stylesheet has to be exactly 'background-color: (r,g,b);'.
        Returns tuple (r,g,b) with dimensionless units (0 - 1).
        """
        stilo = str(widget.styleSheet())
        stilo = stilo.lower()
        for x in ['background-color', ':', 'rgb', '(', ')', ';', ' ']:
            stilo = stilo.replace(x, '')
        rgb = tuple([int(v) for v in stilo.split(',')])
        return rgb


    def setBG(self, widget, rgb):
        """
        Set background color for a widget using stylesheet.
        """
        stilo = 'background-color: rgb({},{},{});'.format(*rgb)
        widget.setStyleSheet(stilo)


    def pickBackground(self):
        """
        Select a solid background color through system dialog.
        """
        color = self.settings['render']['backcolor']
        color = QtGui.QColor.fromRgb(*color)
        color = QColorDialog.getColor(initial=color, parent=self)

        if not color.isValid(): return

        self.showTabs(2)
        color = color.getRgb()[:3]
        print(gogo.timestamp() + 'New background color ' + str(color))
        self.settings['render']['backcolor'] = color
        self.settings['render']['autoback' ] = False
        self.setBG(self.ui.buttonBackground, color)
        self.ui.checkAutoBack.setChecked(False)
        self.showBackdrop()


    def autoBackground(self):
        """
        Show automatic background color pick in the color button.
        """
        self.showTabs(2)
        check = self.ui.checkAutoBack.isChecked()
        self.settings['render']['autoback'] = check
        if check:
            msg = 'Engaging automatic background color pick'
            print(gogo.timestamp() + msg)
            if self.K is None: self.fetchPrimeKey()

            color = algo.bordercolor(self.K)
            self.settings['render']['backcolor'] = color
            self.setBG(self.ui.buttonBackground, color)
            self.refreshPrefs()


    def pickLineColor(self):
        """
        Select line art color through system dialog.
        """
        color = self.settings['render']['linecolor']
        color = QtGui.QColor.fromRgb(*color)
        color = QColorDialog.getColor(initial=color, parent=self)
        if not color.isValid(): return

        self.showTabs(2)
        color = color.getRgb()[:3]
        print(gogo.timestamp() + 'New line art color ' + str(color))
        self.settings['render']['linecolor'] = color
        self.setBG(self.ui.buttonLineColor, color)


    def selectBackdrop(self):
        """
        Select backdrop image through dialog.
        """
        print(gogo.timestamp() + 'Select backdrop file ...', end='')
        f, _ = QFileDialog.getOpenFileName(self, \
               directory = self.keyFolder, \
               caption   = 'MuddyMorph - Select background image', \
               filter    = "Images (*.png *.jpg)\nAll files (*.*)")

        if not path.isfile(f):
            f, name = '', ''
            print('None')
        else:
            name, _ = path.splitext(path.basename(f))
            print(name)           

        self.showTabs(2)
        self.settings['render']['backdrop'] = f
        self.ui.buttonBackdrop.setText(name)
        self.showBackdrop()


    def showBackdrop(self):
        """
        Show background color and/or image.
        """
        print(gogo.timestamp() + 'Showing background and representative key')

        fi = self.settings['render']['backdrop']
        if self.K is None: self.fetchPrimeKey()
        h, w = self.K.shape[:2]

        if fi:
            hi, wi = algo.read_image_dimensions(fi)
            if (hi, wi) != (h, w):
                hdr  = 'MuddyMorph - Image size mismatch'
                msg  = 'Sorry, but the backdrop has to be '
                msg += 'of the exact same size as the key frames'

                self.settings['render']['backdrop'] = ''
                self.ui.buttonBackdrop.setText('Select file ...')
                self.ui.statusBar.showMessage(msg)
                QMessageBox.critical(self, hdr, msg)
                return

            if self.B is None: self.B = algo.load_rgba(fi)

        else:
            self.B = None
            self.ui.buttonBackdrop.setText('Select file ...')

        clr   = self.settings['render']['backcolor'  ]
        trans = self.settings['render']['transparent']
        qual  = self.settings['render']['quality'    ]
        G     = algo.make_background(h, w, clr, trans, self.B)
        C     = algo.composite_bitmaps(self.K, G)
        fo    = path.join(self.settings['temppath'], 'bgc.png')

        algo.save_rgba(C, fo, qual)
        self.settings['render']['backdrop'] = fi
        self.showBitmap(self.ui.previewPrefs, fo)


    def finishPrefs(self, event=None, moveon=True):
        """
        Finish project definition (step 2).
        """
        self.refreshPrefs(showbg=False)

        if moveon:
            print(gogo.timestamp() + 'Moving on to step 3')

            # Vectorize settings
            msg = 'Smearing out settings to match the number of key frames'
            print(gogo.timestamp() + msg)
            gogo.expand_settings(self.settings)

            # Apply typical number of inbetweens to all frames
            tweens = self.ui.spinTypicalTweens.value()
            n = len(self.settings['motion']['inbetweens'])
            self.settings['motion']['inbetweens'] = [tweens] * n

        # Remember the background for faster rendering
        self.G = gogo.background(self.settings, self.K, self.B)

        # Unlock the next step
        if moveon:
            print(gogo.timestamp() + 'Moving on to step 3')
            msg  = 'Excellent. Let us now extract silhouettes through '
            msg += 'crude black and white conversion.'
            self.showTabs(3)
            self.ui.statusBar.showMessage(msg)

        # Select and process the representative key frame
        k = self.ui.tableKey.currentIndex().row()
        self.connectEdge(False)
        self.ui.tableEdge.selectRow(k)
        self.connectEdge(True)
        self.switchEdge()


    def connectEdge(self, connect=True):
        """
        Enable or disable all knob connections for the silhouette tab.
        This serves to prevent an avalanche of signals firing when several
        controls are updated simultaneously after a frame selection.
        """
        if connect:
            self.ui.buttonEdge      .clicked.connect(self.makeEdge  )
            self.ui.buttonAllEdge   .clicked.connect(self.allEdge   )
            self.ui.buttonFinishEdge.clicked.connect(self.finishEdge)
        else:
            self.ui.buttonEdge      .clicked.disconnect(self.makeEdge  )
            self.ui.buttonAllEdge   .clicked.disconnect(self.allEdge   )
            self.ui.buttonFinishEdge.clicked.disconnect(self.finishEdge)


    def collectEdge(self):
        """
        Collect current edge detection settings from the controls.
        """
        k = self.ui.tableEdge.currentIndex().row()
        s = self.settings['edge']

        s['blur'     ][k] = self.ui.spinBlur.value()
        s['threshold'][k] = self.ui.spinThreshold.value()
        s['scharr'   ][k] = self.ui.checkScharr.isChecked()
        s['invert'   ][k] = self.ui.checkInvert.isChecked()
        s['channel'  ][k] = str(self.ui.comboChannel.currentText()).lower()


    def switchEdge(self):
        """
        A new key frame has been selected for contour extraction.
        Apply the settings for this frame, and then create a preview.
        """
        if self.busyBusy(): return

        k = self.ui.tableEdge.currentIndex().row()
        s = self.settings['edge']

        self.connectEdge(False)
        self.ui.spinBlur     .setValue  (s['blur'     ][k])
        self.ui.spinThreshold.setValue  (s['threshold'][k])
        self.ui.checkScharr  .setChecked(s['scharr'   ][k])
        self.ui.checkInvert  .setChecked(s['invert'   ][k])

        t = s['channel'][k].capitalize()
        i = self.ui.comboChannel.findText(t)
        if i != -1: self.ui.comboChannel.setCurrentIndex(i)

        self.makeEdge(recycle=True)
        self.connectEdge(True)


    def makeEdge(self, recycle=False):
        """
        Perform silhouette extraction and contour detection.
        Show the result in the preview pane.
        """

        k = self.ui.tableEdge.currentIndex().row()
        f = path.join(self.settings['temppath'],
                      'k{0:03d}'.format(k + 1), 'shape.png')

        self.showTabs(3)
        self.collectEdge()

        if recycle and path.isfile(f):
            self.ui.statusBar.showMessage('')
            self.showBitmap(self.ui.previewEdge, f)
        else:
            msg = 'Extracting silhouette for key {} ...'
            self.enableButtons(False)
            self.ui.statusBar.showMessage(msg.format(k + 1))
            self.showBitmap(self.ui.previewEdge, 'wait')

            if self.gridX is None: self.fetchPrimeKey()

            self.comput = ThreadEdge(self.settings, k, \
                X=self.gridX, Y=self.gridY, \
                showsil  = self.ui.checkShowSilhouette.isChecked(), \
                showedge = self.ui.checkShowEdge      .isChecked(), \
                showcom  = self.ui.checkShowCenter    .isChecked())

            self.comput.finished.connect(self.madeEdge)
            self.comput.crashed.connect(self.reportError)
            self.comput.start()


    def madeEdge(self):
        """
        Silhouete extraction has finished. Show that juicy result.
        """
        if self.comput:
            self.ui.statusBar.showMessage(self.comput.report)
            self.showBitmap(self.ui.previewEdge, self.comput.chart)
        self.enableButtons(True)


    def allEdge(self):
        """
        Apply current silhouette extraction settings to all keys.
        This involves destroying all intermediate results for other frames.
        """
        self.collectEdge()

        # Make an announcement
        k   = self.ui.tableEdge.currentIndex().row()
        n   = len(self.settings['keyframes'])
        msg = 'Applying silhouette settings of key {} to all'
        print(gogo.timestamp() + msg.format(k + 1))

        # Apply settings
        for prop in self.settings['edge']:
            v = self.settings['edge'][prop][k]
            self.settings['edge'][prop] = [v] * n

        # Scrap temporary files
        scrap = [i for i in range(n) if not i == k]
        gogo.clear_temp_key(self.settings, scrap)

        # Done!
        msg = 'Current silhouette extraction settings were applied to all keys'
        self.ui.statusBar.showMessage(msg)


    def finishEdge(self, event=None, moveon=True):
        """
        Done with silhouette extraction. Time to move on!
        """
        if moveon:
            if moveon: print(gogo.timestamp() + 'Moving on to step 4')

            # Best get our shit together
            self.collectEdge()

            # Delete lingering keypoint detection files
            # (Can happen if the user came from the future and changed the past)
            gogo.clear_temp_traject(self.settings)

            # Unlock the next step
            print(gogo.timestamp() + 'Moving on to step 4')
            self.showTabs(4)
            msg = 'Now we are ready for defining trajectories.'
            self.ui.statusBar.showMessage(msg)

        # Select and process the representative morph
        k = self.ui.tableKey.currentIndex().row()
        m = algo.np.clip(k, 0, gogo.count_morphs(self.settings))
        self.connectTraject(False)
        self.ui.tableTraject.selectRow(m)
        self.connectTraject(True)
        self.switchTraject()


    def connectTraject(self, connect=True):
        """
        Enable or disable all knob connections for the trajectories tab.
        """
        if connect:
            self.ui.checkSilhouette    .clicked.connect(self.arcTraject   )
            self.ui.checkArc           .clicked.connect(self.arcTraject   )
            self.ui.checkSpin          .clicked.connect(self.arcTraject   )
            self.ui.buttonTraject      .clicked.connect(self.makeTraject  )
            self.ui.buttonAllTraject   .clicked.connect(self.allTraject   )
            self.ui.buttonFinishTraject.clicked.connect(self.finishTraject)
        else:
            self.ui.checkSilhouette    .clicked.disconnect(self.arcTraject   )
            self.ui.checkArc           .clicked.disconnect(self.arcTraject   )
            self.ui.checkSpin          .clicked.disconnect(self.arcTraject   )
            self.ui.buttonTraject      .clicked.disconnect(self.makeTraject  )
            self.ui.buttonAllTraject   .clicked.disconnect(self.allTraject   )
            self.ui.buttonFinishTraject.clicked.disconnect(self.finishTraject)


    def arcTraject(self):
        """
        Enable or disable the arc related check boxes.
        """
        can_arc  = self.ui.checkSilhouette.isChecked()
        can_spin = self.ui.checkArc.isChecked() and can_arc

        self.ui.checkArc .setEnabled(can_arc )
        self.ui.checkSpin.setEnabled(can_spin)

        if not can_arc : self.ui.checkArc .setChecked(False)
        if not can_spin: self.ui.checkSpin.setChecked(False)


    def collectTraject(self):
        """
        Collect current trajectory detection settings from the controls.
        """
        m    = self.ui.tableTraject.currentIndex().row()
        st   = self.settings['traject']
        se   = self.settings['edge'   ]
        a, b = gogo.morph_key_indices(self.settings, m)

        st['corners'      ][m] = self.ui.checkCorners   .isChecked()
        st['silhouette'   ][m] = self.ui.checkSilhouette.isChecked()
        st['arc'          ][m] = self.ui.checkArc       .isChecked()
        st['spin'         ][m] = self.ui.checkSpin      .isChecked()

        st['detail'       ][m] = self.ui.spinDetail     .value()
        st['maxpoints'    ][m] = self.ui.spinMaxPoints  .value()
        st['neighbours'   ][m] = self.ui.spinNeighbours .value()
        st['similim'      ][m] = self.ui.spinSimiLim    .value()
        st['maxmove'      ][m] = self.ui.spinMaxMove    .value()

        se['cornercatcher'][a] = str(self.ui.comboCornerCatcher.currentText())
        se['cornercatcher'][b] = str(self.ui.comboCornerCatcher.currentText())


    def switchTraject(self):
        """
        Another trajectory visualization has been selected.
        Apply the settings for the selected morph, and then create a preview.
        """
        if self.busyBusy(): return

        # Look up settings
        m      = self.ui.tableTraject.currentIndex().row()
        st     = self.settings['traject']
        se     = self.settings['edge'   ]
        a, b   = gogo.morph_key_indices(self.settings, m)
        
        # Spinning only makes sense in case of arcs
        if not st['arc'][m]: st['spin'][m] = False

        # Apply settings
        self.connectTraject(False)
        self.ui.spinMaxPoints  .setValue  (st['maxpoints' ][m])
        self.ui.spinNeighbours .setValue  (st['neighbours'][m])
        self.ui.checkCorners   .setChecked(st['corners'   ][m])
        self.ui.checkSilhouette.setChecked(st['silhouette'][m])
        self.ui.checkArc       .setChecked(st['arc'       ][m])
        self.ui.checkSpin      .setChecked(st['spin'      ][m])
        self.ui.spinSimiLim    .setValue  (st['similim'   ][m])
        self.ui.spinMaxMove    .setValue  (st['maxmove'   ][m])
        self.ui.spinDetail     .setValue  (st['detail'    ][m])

        t = se['cornercatcher'][a]
        i = self.ui.comboCornerCatcher.findText(t)
        if i != -1: self.ui.comboCornerCatcher.setCurrentIndex(i)

        self.makeTraject(recycle=True)
        self.connectTraject(True)


    def allTraject(self):
        """
        Apply trajectory detection settings of this morph to all sequences.
        """
        if self.busyBusy(): return
        self.collectTraject()

        # Make an announcement
        m    = self.ui.tableTraject.currentIndex().row()
        n_k  = len(self.settings['keyframes'])
        n_m  = gogo.count_morphs(self.settings)
        a, b = gogo.morph_key_indices(self.settings, m)
        msg  = 'Applying trajectory settings of morph {} to all'
        print(gogo.timestamp() + msg.format(m + 1))

        # Apply settings
        for prop in self.settings['traject']:
            v = self.settings['traject'][prop][m]
            self.settings['traject'][prop] = [v] * n_m

        # Scrap temporary files not associated with the current morph
        scrap_m = [i for i in range(n_m) if not i == m]
        scrap_k = [i for i in range(n_k) if not i in (a, b)]
        gogo.clear_temp_traject(self.settings, m=scrap_m, k=scrap_k)

        # Done!
        msg = 'Current trajectory detection settings were applied to all morphs'
        self.ui.statusBar.showMessage(msg)


    def makeTraject(self, recycle=False):
        """
        Fire up the trajectory detection thread.
        """

        # Abort the mission?
        if self.busyBusy():
            if not self.settings['step'] == 4: return
            self.comput.abort = True
            self.ui.statusBar.showMessage('Aborting ...')
            print(gogo.timestamp() + 'Abort traject mission!')
            return

        # Shorthand name for current traject alias morph number
        m = self.ui.tableTraject.currentIndex().row()

        # Show that something is cooking in this kitchen
        self.enableButtons(False)
        self.showTabs(4)
        self.showBitmap(self.ui.previewTraject, 'wait')
        msg = 'Detecting trajectories for sequence {} ...'
        self.ui.statusBar.showMessage(msg.format(m + 1))
        self.collectTraject()


        # Double check we have a mesh grid and background image on board
        if self.gridX is None:
            self.fetchPrimeKey()
        if self.G is None:
            self.G = gogo.background(self.settings, self.K, self.B)

        self.stopwatch = -time()

        self.comput = ThreadTraject(self.settings, m, recycle=recycle,
                                    X=self.gridX, Y=self.gridY, G=self.G)

        self.comput.report  .connect(self.progressTraject)
        self.comput.crashed .connect(self.reportError)
        self.comput.finished.connect(self.madeTraject)
        self.comput.start()


    def progressTraject(self, report=''):
        """
        A trajectory progress report has arrived.
        This is either an image file name or status message.
        """
        if path.isfile(report) or report == 'wait':
            self.showBitmap(self.ui.previewTraject, report)
        else:
            self.ui.statusBar.showMessage(report)


    def madeTraject(self):
        """
        The traject thread has finished.
        Store the result, and carry on with business as usual.
        """
        self.stopwatch += time()
        if self.comput and self.comput.abort:
            msg = 'Traject detection aborted'
        else:
            msg = 'Traject detection took ' + gogo.duration(self.stopwatch)
        self.ui.statusBar.showMessage(msg)
        self.enableButtons(True)


    def finishTraject(self, event=None, moveon=True):
        """
        Finish blob definition (step 4).
        """
        if self.busyBusy(): return

        if moveon:
            print(gogo.timestamp() + 'Moving on to step 5')

            # Delete lingering inbetweens
            # (Can happen if the user hops back and forth through the tabs)
            gogo.clear_temp_motion(self.settings)

            # Unlock the next step
            msg = 'Finally, we are ready to make some moves.'
            self.showTabs(5)
            self.ui.statusBar.showMessage(msg)

        # Select and process the representative morph sequence
        m = self.ui.tableTraject.currentIndex().row()
        fps = self.settings['render']['framerate']
        self.connectMotion(False)
        self.ui.spinFrameRate.setValue(fps)
        self.ui.tableMotion.selectRow(m)
        self.connectMotion(True)
        self.switchMotion()


    def connectMotion(self, connect=True):
        """
        Enable or disable all knob connections for the motion tab.
        """
        try:
            if connect:
                self.ui.checkBlob          .clicked.connect(self.blobMotion   )
                self.ui.buttonPlayMotion   .clicked.connect(self.toggleMotion )
                self.ui.buttonRevPrev      .clicked.connect(self.previewMotion)
                self.ui.buttonMotion       .clicked.connect(self.makeMotion   )
                self.ui.buttonAllMotion    .clicked.connect(self.allMotion    )
                self.ui.buttonFinishMotion .clicked.connect(self.finishMotion )
                self.ui.spinFrameRate.valueChanged .connect(self.speedMotion  )
                self.ui.slideMotion  .valueChanged .connect(self.hopMotion    )
            else:
                self.ui.checkBlob          .clicked.disconnect(self.blobMotion   )
                self.ui.buttonPlayMotion   .clicked.disconnect(self.toggleMotion )
                self.ui.buttonRevPrev      .clicked.disconnect(self.previewMotion)
                self.ui.buttonMotion       .clicked.disconnect(self.makeMotion   )
                self.ui.buttonAllMotion    .clicked.disconnect(self.allMotion    )
                self.ui.buttonFinishMotion .clicked.disconnect(self.finishMotion )
                self.ui.spinFrameRate.valueChanged .disconnect(self.speedMotion  )
                self.ui.slideMotion  .valueChanged .disconnect(self.hopMotion    )
        except TypeError as e:
            # Do not crash immediately, self-repair may be around the corner.
            msg = 'Watch out! connectMotion issue: '
            print(gogo.timestamp() + msg + str(e))


    def blobMotion(self):
        """
        Enable the blob shape controls only when appropriate.
        """
        enable = self.ui.checkBlob.isChecked()
        self.ui.spinHardness.setEnabled(enable)
        self.ui.spinInflate .setEnabled(enable)


    def collectMotion(self):
        """
        Collect current motion making settings from the controls.
        """
        m  = self.ui.tableMotion.currentIndex().row()
        sm = self.settings['motion']
        sr = self.settings['render']

        sm['inbetweens' ][m] = self.ui.spinBetweens .value()
        sm['fade'       ][m] = self.ui.spinFade     .value()
        sm['blob'       ][m] = self.ui.checkBlob    .isChecked()
        sm['profile'    ][m] = str(self.ui.comboMotion.currentText()).lower()

        sr['framerate'     ] = self.ui.spinFrameRate.value()
        sr['vignette'      ] = self.ui.spinVignette .value()
        sr['blobscale'     ] = self.ui.spinInflate  .value()
        sr['blobhardness'  ] = self.ui.spinHardness .value()
        sr['reversepreview'] = self.ui.buttonRevPrev.isChecked()


    def switchMotion(self):
        """
        A new sequence has been selected for inbetweening.
        Apply the settings for the selected morph, and then create a preview.
        """
        if self.busyBusy(): return

        m    = self.ui.tableMotion.currentIndex().row()
        sm   = self.settings['motion']
        sr   = self.settings['render']
        blob = sm['blob'][m]

        self.connectMotion(False)

        self.ui.spinFade     .setValue(sm['fade'][m])
        self.ui.spinBetweens .setValue(sm['inbetweens'][m])

        self.ui.checkBlob    .setChecked(blob)
        self.ui.spinInflate  .setEnabled(blob)
        self.ui.spinHardness .setEnabled(blob)

        self.ui.spinVignette .setValue  (sr['vignette'      ])
        self.ui.spinInflate  .setValue  (sr['blobscale'     ])
        self.ui.spinHardness .setValue  (sr['blobhardness'  ])
        self.ui.buttonRevPrev.setChecked(sr['reversepreview'])

        t = sm['profile'][m].capitalize()
        i = self.ui.comboMotion.findText(t)
        if i != -1: self.ui.comboMotion.setCurrentIndex(i)

        self.connectMotion(True)
        self.makeMotion(recycle=True)


    def allMotion(self):
        """
        Apply current settings to all morphs.
        """
        if self.busyBusy(): return
        self.collectMotion()

        m = self.ui.tableMotion.currentIndex().row()
        n = gogo.count_morphs(self.settings)

        # Make an announcement
        print('Applying motion settings for sequence {} to all'.format(m + 1))

        # Apply settings
        for prop in self.settings['motion']:
            v = self.settings['motion'][prop][m]
            self.settings['motion'][prop] = [v] * n

        # Scrap temporary files not associated with the current morph
        scrap = [i for i in range(n) if not i == m]
        gogo.clear_temp_motion(self.settings, scrap)

        # Done!
        msg = 'Current motion settings were applied to all morphs'
        self.ui.statusBar.showMessage(msg)


    def makeMotion(self, recycle=False):
        """
        Make inbetween frames for the selected morph sequence.
        """

        # Abort the mission?
        if self.busyBusy():
            if not self.settings['step'] == 5: return
            self.comput.abort = True
            self.ui.statusBar.showMessage('Aborting ...')
            print(gogo.timestamp() + 'Abort morph mission!')
            return

        # Get those settings
        self.showTabs(5)
        self.collectMotion()

        # Show that something is cooking in this kitchen
        self.enableButtons(False)
        self.showBitmap(self.ui.previewMotion, 'wait')

        # Destroy previous results. Cleanliness above all.
        m = self.ui.tableMotion.currentIndex().row()
        if not recycle: gogo.clear_temp_motion(self.settings, [m])
        
        # Reset the frame slider
        imax = self.settings['motion']['inbetweens'][m] + 1
        self.movieReel = []
        self.ui.slideMotion.setValue(0)
        self.ui.slideMotion.setMaximum(imax)        

        # Fire it up!
        self.stopwatch = -time()

        self.comput = ThreadMotion(self.settings, m, recycle,
                                   X=self.gridX, Y=self.gridY, G=self.G)

        self.comput.report  .connect(self.progressMotion)
        self.comput.crashed .connect(self.reportError   )
        self.comput.finished.connect(self.madeMotion    )
        self.comput.start()


    def progressMotion(self, report=''):
        """
        A fresh inbetween has been generated,
        so better show it in the preview pane.
        Also deduce the current frame index from the image file name.
        """
        if path.isfile(report):
            name = path.basename(report)
            try:
                i = int(name[1:4]) - 1
                self.ui.slideMotion.setValue(i)
            except ValueError:
                pass
            self.showBitmap(self.ui.previewMotion, report)
            
        elif report == 'wait':
            self.showBitmap(self.ui.previewMotion, report)
            
        else:
            self.ui.statusBar.showMessage(report)


    def madeMotion(self):
        """
        The morph sequence is finished. Show it a a looped animation.
        """
        self.stopwatch += time()

        # Prepare movie playback
        if self.comput \
                 and not self.comput.abort \
                 and not self.comput.movie is None:
            self.movieReel = []
            for f in range(len(self.comput.movie)):
                msg = 'Preparing morph movie frame {}'.format(f + 1)
                print(gogo.timestamp() + msg)
                self.ui.statusBar.showMessage(msg)

                pix = QtGui.QPixmap(self.comput.movie[f])
                pix = pix.scaled(self.ui.previewMotion.size(),
                                 QtCore.Qt.KeepAspectRatio,
                                 QtCore.Qt.SmoothTransformation)
                self.movieReel.append(pix)
            self.previewMotion()

        # Hint at the blood and tears shed by our hard working algorithms
        if self.comput and self.comput.abort:
            msg = 'Morph was aborted prematurely'
        else:
            msg = 'Morphing took ' + gogo.duration(self.stopwatch)
        self.ui.statusBar.showMessage(msg)

        # Hand the steering wheel back to the user
        self.enableButtons(True)


    def previewMotion(self):
        """
        Start the morph preview movie (if any).
        """
        if not len(self.movieReel):
            msg = 'Preview request cannot be honoured '
            msg += 'due to empty movie reel ...'
            print(gogo.timestamp() + msg)
            return

        # Make sure we have up to date playback parameters
        rev = self.ui.buttonRevPrev.isChecked()
        fps = self.ui.spinFrameRate.value()
        t   = int(1000. / fps)
        s   = self.settings['render']
        s['framerate'] = fps
        s['reversepreview'] = rev

        # Rewind when the crowd says Bo Selecta?
        self.movieSequence = list(range(len(self.movieReel)))
        if rev:
            self.movieSequence += list(range(len(self.movieReel) - 2, 0, -1))
        
        # Make sure the frame slider is in a decent position
        imax = len(self.movieReel) - 1
        if self.ui.slideMotion.value() > imax:
            self.ui.slideMotion.setValue(0)
        self.ui.slideMotion.setMaximum(imax)

        # (Re)start the show
        self.movieTimer.stop()
        msg = 'Starting playback loop: {} frames, reverse {}, {} fps'
        gogo.shoutout(msg.format(len(self.movieSequence), rev, fps))
        self.ui.buttonPlayMotion.setText(u'Stop  ■')
        self.movieTimer.start(t, self)


    def timerEvent(self, event=None):
        """
        Show the next frame of the current morph movie.
        """
        if not len(self.movieReel):
            self.movieTimer.stop()
            return

        # Move on to the next frame
        self.movieIndex += 1
        if self.movieIndex >= len(self.movieSequence):
            self.movieIndex = 0

        # Show the frame in all of its glory
        f = self.movieSequence[self.movieIndex]
        pixmap = self.movieReel[f]
        self.ui.slideMotion.setValue(f)
        self.ui.previewMotion.setPixmap(pixmap)


    def toggleMotion(self):
        """
        The play/pause button has been pressed.
        Appropriate action depends heavily on the circumstances.
        """
        if self.movieTimer.isActive():
            # Pause movie playback
            print(gogo.timestamp() + 'Pausing movie playback')
            self.ui.buttonPlayMotion.setText(u'Play  ►')
            self.movieTimer.stop()

        elif len(self.movieReel) > 0:
            # Continue movie playback
            self.previewMotion()

        else:
            # Generate inbetweens and start movie
            self.makeMotion()


    def speedMotion(self):
        """
        Change the playback speed (fps) while the preview is running.
        """
        if not self.movieTimer.isActive(): return

        fps = self.ui.spinFrameRate.value()
        t = int(1000. / fps)
        self.settings['render']['framerate'] = fps
        gogo.shoutout('Changing playback speed to {}'.format(fps))

        self.movieTimer.stop()
        self.movieTimer.start(t, self)


    def hopMotion(self):
        """
        Show the frame that was just selected through the preview slider.
        """
        if self.movieTimer.isActive() or not len(self.movieReel): return

        self.movieIndex = self.ui.slideMotion.value()
        f = self.movieSequence[self.movieIndex]
        pixmap = self.movieReel[f]
        self.ui.previewMotion.setPixmap(pixmap)


    def stopMotion(self):
        """
        Clear the current movie preview (if any).
        """
        self.movieTimer.stop()

        self.movie         = None
        self.movieReel     = []
        self.movieSequence = []
        self.movieIndex    = -1

        self.showBitmap(self.ui.previewMotion)


    def finishMotion(self, event=None, moveon=True):
        """
        Finish motion tweaking (step 6).
        """

        # Collect motion settings
        self.collectMotion()

        # Apply user preferences for rendering
        s = self.settings['render']
        if not path.isdir(s['folder']):
            s['folder'] = path.expanduser('~')
        if not s['name']:
             s['name'] = 'morph'
        self.ui.editFolder  .setText   (s['folder' ])
        self.ui.radioReverse.setChecked(s['reverse'])

        # Unlock the next and final step
        if not moveon: return
        print(gogo.timestamp() + 'Moving on to step 7')
        msg = 'And now we are ready to harvest the fruits of our hard labour.'
        self.showTabs(6)
        self.ui.statusBar.showMessage(msg)


    def selectFolder(self):
        """
        Select an export folder through popup dialog.
        """
        print(gogo.timestamp() + 'Select an export folder ... ', end='')

        folder = self.ui.editFolder.text()
        folder = str(QFileDialog.getExistingDirectory(self,
                     directory = folder, \
                     caption   = 'MuddyMorph - Select export folder'))

        print(folder)
        if path.isdir(folder):
            self.ui.editFolder.setText(folder)


    def startRender(self):
        """
        Render the full project, and export the frames to the given folder.
        """

        # Verify a valid folder has been selected
        if not path.isdir(self.settings['render']['folder']):
            hdr = 'MuddyMorph - Invalid folder'
            msg = 'Please select a valid export folder'
            QMessageBox.critical(self, hdr, msg)
            return

        # Collect user preferences
        self.settings['step'  ] = 6
        self.settings['render']['folder' ] = str(self.ui.editFolder.text())
        self.settings['render']['name'   ] = str(self.ui.editName.text())
        self.settings['render']['reverse'] = self.ui.radioReverse.isChecked()
        gogo.save_settings(self.settings, self.settingsfile)

        # Disable all but the abort button
        self.enableButtons(False)

        # Fire up that lean and mean rendering machine
        self.comput = ThreadRender(self.settings)
        self.comput.report  .connect(self.progressRender)
        self.comput.crashed .connect(self.reportError   )
        self.comput.finished.connect(self.finishRender  )
        self.comput.start()


    def abortRender(self):
        """
        Abandon ship! Women and children first!
        The user wants to abort the current render job.
        """
        if not self.busyBusy(): return
        if not self.settings['step'] == 6: return

        self.ui.statusBar.showMessage('Aborting ...')
        self.comput.abort = True


    def progressRender(self, report=''):
        """
        A fresh inbetween has been generated,
        so better show it in the preview pane.
        """
        if path.isfile(report) or report == 'wait':
            self.showBitmap(self.ui.previewRender, report)
        else:
            self.ui.statusBar.showMessage(report)


    def finishRender(self):
        """
        This is the very end of the wizard line. Time to say goodbye!
        """
        if self.comput and self.comput.abort:
            self.ui.statusBar.showMessage('Render was aborted')
        self.enableButtons()


class TableModel(QtCore.QAbstractTableModel):
    """
    Class required for creating Qt table content.
    Copied from old post on www.saltycrane.com (not available anymore).
    """
    def __init__(self, tabledata, tableheader, parent=None, *args):
        QtCore.QAbstractTableModel.__init__(self, parent, *args)
        self.tabledata = tabledata
        self.tableheader = tableheader

    def rowCount(self, parent):
        return len(self.tabledata)

    def columnCount(self, parent):
        if len(self.tabledata) == 0:
            return 0
        else:
            return len(self.tabledata[0])

    def data(self, index, role):
        if not index.isValid():
            return None
        elif role != QtCore.Qt.DisplayRole:
            return None
        else:
            row = index.row()
            col = index.column()
            return self.tabledata[row][col]

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and \
                role == QtCore.Qt.DisplayRole:
            return self.tableheader[col]
        return None


class ThreadEdge(QtCore.QThread):
    """
    Extract silhouette contours from the selected key frame.

    This has been made a separate (non-interactive) thread for the sole
    purpose of preventing the GUI from freezing. Since this step usually does
    not take long (a few seconds tops) the thread cannot be aborted externally.
    """
    crashed = QtCore.pyqtSignal(object, object)

    def __init__(self, settings, k,
                 showsil=True, showedge=True, showcom=True, X=None, Y=None):

        super().__init__()

        self.settings = settings
        self.k        = k
        self.showsil  = showsil
        self.showedge = showedge
        self.showcom  = showcom
        self.X        = X
        self.Y        = Y
        self.report   = ''
        self.chart    = 'missing'

    def run(self):
        msg = 'Silhouette extraction for key {}'.format(self.k + 1)
        print(gogo.timestamp() + msg)
        stopwatch = -time()
        try:
            f = gogo.silhouette(self.settings, self.k   ,
                                X        = self.X       ,
                                Y        = self.Y       ,
                                showsil  = self.showsil ,
                                showedge = self.showedge,
                                showcom  = self.showcom )[0]
        except:
            self.crashed.emit(sys.exc_info(), traceback.format_exc())
            return

        stopwatch  += time()
        self.report = 'Silhouette extraction took ' + gogo.duration(stopwatch)
        self.chart  = f


class ThreadTraject(QtCore.QThread):
    """
    Extract corner keypoints, detect trajectories, and make a midpoint warp.
    """
    report  = QtCore.pyqtSignal(str)
    crashed = QtCore.pyqtSignal(object, object)

    def __init__(self, settings, m, recycle=False,
                 X=None, Y=None, G=None, midpoint=False):

        super().__init__()

        self.settings = settings
        self.m        = m
        self.X        = X
        self.Y        = Y
        self.G        = G
        self.recycle  = recycle
        self.midpoint = midpoint
        self.abort    = False


    def run(self):
        # Distill trajectories
        try:
            result = gogo.trajectory(self.settings, self.m,
                                     self.recycle, self, self.X, self.Y)
            if result is None:
                self.abort = True
                return
            nodes, Ka, Kb, com_a, com_b = result
        except:
            self.crashed.emit(sys.exc_info(), traceback.format_exc())
            return

        # Midpoint warp
        if not self.midpoint: return
        try:
            fm = gogo.midpoint(self.settings, self.m, recycle=self.recycle,
                               Ka=Ka, Kb=Kb, G=self.G, nodes=nodes,
                               com_a=com_a, com_b=com_b)
            self.report.emit(fm)
        except:
            self.crashed.emit(sys.exc_info(), traceback.format_exc())


class ThreadMotion(QtCore.QThread):
    """
    Render inbetween frames for a particular morph sequence.
    """
    report  = QtCore.pyqtSignal(str)
    crashed = QtCore.pyqtSignal(object, object)

    def __init__(self, settings, m, recycle=False, X=None, Y=None, G=None):

        super().__init__()

        self.settings = settings
        self.recycle  = recycle
        self.m        = m
        self.X        = X
        self.Y        = Y
        self.G        = G
        self.movie    = None
        self.abort    = False


    def run(self):
        try:
            self.movie = gogo.motion(self.settings, self.m,
                                     recycle_frames=self.recycle, thread=self,
                                     X=self.X, Y=self.Y, G=self.G)

        except:
            self.crashed.emit(sys.exc_info(), traceback.format_exc())


class ThreadRender(QtCore.QThread):
    """
    Render the whole project. Meep meep!
    """
    report  = QtCore.pyqtSignal(str)
    crashed = QtCore.pyqtSignal(object, object)

    def __init__(self, settings, recycle=True):

        super().__init__()

        self.settings = settings
        self.recycle  = recycle
        self.abort    = False


    def run(self):
        try:
            gogo.render(self.settings, self.recycle, self)
        except:
            self.crashed.emit(sys.exc_info(), traceback.format_exc())


def main(app=None, parent=None):
    """
    Start GUI.
    """

    print("\n\nMuddyMorph Wizard bids you welcome!")
    print("Version {}, Revision {}\n".format(__version__, __revision__))

    modal = not app is None

    if not modal: app = QApplication(sys.argv)

    win = Gui(app, parent)
    win.show()

    if not modal: app.exec_()


# Run Forest, run!
if __name__ == "__main__": main()
