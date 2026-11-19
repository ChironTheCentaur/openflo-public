; OpenFlo Windows installer (NSIS 3).
;
; A TEMPLATE: scripts/build_windows_installer.py fills in each placeholder
; written between double at-signs and runs makensis on the result. Do not
; run makensis on this file directly.
;
; What it installs, for the current user only (no administrator rights):
;   %LOCALAPPDATA%\Programs\OpenFlo\python\   a private Windows Python with
;                                             OpenFlo and its dependencies
;   %LOCALAPPDATA%\Programs\OpenFlo\OpenFlo.ico
;   %LOCALAPPDATA%\Programs\OpenFlo\Uninstall OpenFlo.exe
; plus a Start-menu shortcut, an optional Desktop shortcut and an entry in
; Settings > Apps (HKCU). Running a newer installer replaces the python folder
; in place. The user's settings live in %USERPROFILE%\.openflo and are never
; touched, by an upgrade or by the uninstaller.
;
; Before installing, a Privacy page shows the privacy policy's desktop part,
; links to the whole policy and has a "Check for updates when OpenFlo starts"
; box (on by default; /NOUPDATECHECK turns it off for a silent install). The
; answer goes to HKCU\Software\OpenFlo, CheckUpdatesOnStart, which OpenFlo
; reads as its default until the user changes the setting in Preferences.
;
; File types, for the current user: double-clicking an OpenFlo session
; (.flowsession) or project archive (.flowproject) opens it in OpenFlo (the
; window opens a file named on its command line). .fcs is left to whatever
; program owns it. The uninstaller removes both, and an extension only while
; it still points at OpenFlo.

Unicode true
Target amd64-unicode
SetCompressor /SOLID lzma
SetCompressorDictSize 64

!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "nsDialogs.nsh"
!include "FileFunc.nsh"

!define APP_NAME "OpenFlo"
!define APP_VERSION "@@VERSION@@"
!define GUI_MODULE "@@GUI_MODULE@@"
!define PAYLOAD "@@PAYLOAD_DIR@@"
!define MARKER "@@MARKER_NAME@@"
!define UNINSTALLER "Uninstall OpenFlo.exe"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\OpenFlo"
!define APP_KEY "Software\OpenFlo"
!define PRIVACY_URL "@@PRIVACY_URL@@"
; The installer's answer to "Check for updates when OpenFlo starts", read by
; openflo.prefs (INSTALLER_KEY, INSTALLER_UPDATES_VALUE) as the default until
; the user sets it in Preferences. Never written to the user's settings folder.
!define UPDATES_VALUE "CheckUpdatesOnStart"
; The "Desktop application" part of the privacy policy (the policy of
; 17 November 2026), shown on the Privacy page.
!define POLICY_TEXT `The desktop application is open-source software that runs on your computer and sends no analytics or usage data to us.$\r$\n$\r$\nUnless you turn it off in Preferences, it makes a quiet request to GitHub when it starts to see whether a newer release exists. If you choose Update, it downloads the new version from GitHub or the Python Package Index (PyPI), which see your IP address under their own policies. GitHub receives your IP address and the request, under GitHub's own privacy statement. Nothing about your files is sent.$\r$\n$\r$\nNearby send (File > Nearby send…) works as described under "Nearby send" in the policy. Receiving is off until you turn it on. The computer's Nearby name ("Computer" plus four characters unless you rename it), its keys and its list of paired devices are kept in your user folder and never sent to us.$\r$\n$\r$\nReports you choose to send from the app work as described in Section 6 of the policy.`
!define CLASSES "Software\Classes"

Name "${APP_NAME} ${APP_VERSION}"
OutFile "@@OUTFILE@@"
InstallDir "$LOCALAPPDATA\Programs\OpenFlo"
InstallDirRegKey HKCU "${APP_KEY}" "InstallDir"
RequestExecutionLevel user
ShowInstDetails show
ShowUninstDetails show

VIProductVersion "@@VI_VERSION@@"
VIAddVersionKey "ProductName" "${APP_NAME}"
VIAddVersionKey "ProductVersion" "${APP_VERSION}"
VIAddVersionKey "FileVersion" "${APP_VERSION}"
VIAddVersionKey "FileDescription" "${APP_NAME} ${APP_VERSION} setup"
VIAddVersionKey "CompanyName" "@@PUBLISHER@@"
VIAddVersionKey "LegalCopyright" "@@COPYRIGHT@@. MIT License."

!define MUI_ICON "${PAYLOAD}\OpenFlo.ico"
!define MUI_UNICON "${PAYLOAD}\OpenFlo.ico"
!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "Start OpenFlo now"
!define MUI_FINISHPAGE_RUN_FUNCTION LaunchOpenFlo

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "@@LICENSE_FILE@@"
Page custom PrivacyPage PrivacyPageLeave
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Var CheckUpdates
Var CheckUpdatesBox

; The update check is on unless the Privacy page's box is unticked, or a
; silent install is run with /NOUPDATECHECK.
Function .onInit
  SetShellVarContext current
  StrCpy $CheckUpdates 1
  ${GetParameters} $R0
  ClearErrors
  ${GetOptions} $R0 "/NOUPDATECHECK" $R1
  ${IfNot} ${Errors}
    StrCpy $CheckUpdates 0
  ${EndIf}
FunctionEnd

Function un.onInit
  SetShellVarContext current
FunctionEnd

; Before anything is installed: the privacy policy's desktop part, the
; option to turn the update check off, and a link to the whole policy.
Function PrivacyPage
  !insertmacro MUI_HEADER_TEXT "Privacy" "What OpenFlo sends over the internet"
  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 10u "From the OpenFlo privacy policy, Desktop application:"
  Pop $0
  nsDialogs::CreateControl EDIT ${DEFAULT_STYLES}|${WS_TABSTOP}|${ES_MULTILINE}|${ES_READONLY}|${WS_VSCROLL}|${ES_AUTOVSCROLL} ${WS_EX_CLIENTEDGE} 0 11u 100% 64u `${POLICY_TEXT}`
  Pop $0
  ${NSD_CreateCheckbox} 0 79u 100% 12u "Check for updates when OpenFlo starts"
  Pop $CheckUpdatesBox
  ${If} $CheckUpdates == 1
    ${NSD_Check} $CheckUpdatesBox
  ${EndIf}
  ${NSD_CreateLabel} 0 93u 100% 28u "You can change this later in Edit > Preferences > Updates. Problem reports and Nearby send are used only when you choose them."
  Pop $0
  ${NSD_CreateLink} 0 124u 100% 12u "${PRIVACY_URL}"
  Pop $0
  ${NSD_OnClick} $0 OpenPrivacyPolicy
  nsDialogs::Show
FunctionEnd

Function PrivacyPageLeave
  ${NSD_GetState} $CheckUpdatesBox $0
  ${If} $0 == ${BST_CHECKED}
    StrCpy $CheckUpdates 1
  ${Else}
    StrCpy $CheckUpdates 0
  ${EndIf}
FunctionEnd

Function OpenPrivacyPolicy
  Pop $0
  ExecShell "open" "${PRIVACY_URL}"
FunctionEnd

; Every start (both shortcuts and "Start OpenFlo now") runs in the user's
; home folder, so the first Add FCS / Open dialog opens there and not in the
; install folder. -P keeps that working folder off Python's import path.
Function LaunchOpenFlo
  SetOutPath "$PROFILE"
  Exec '"$INSTDIR\python\pythonw.exe" -P -m ${GUI_MODULE}'
FunctionEnd

; Move the python folder to python.old. Windows refuses to rename a folder
; while a program inside it is running, so this fails while OpenFlo is open:
; the user is asked to close it, and Cancel (or a silent run) stops before
; anything is deleted, rather than leaving a half-deleted install behind.
!macro MOVE_RUNTIME_ASIDE UN
Function ${UN}MoveRuntimeAside
  RMDir /r "$INSTDIR\python.old"
  retry:
  ClearErrors
  Rename "$INSTDIR\python" "$INSTDIR\python.old"
  ${If} ${Errors}
    MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION "OpenFlo is still running. Close every OpenFlo window, then choose Retry." /SD IDCANCEL IDRETRY retry
    Abort "Stopped: OpenFlo is still running. Nothing was changed."
  ${EndIf}
FunctionEnd
!macroend
!insertmacro MOVE_RUNTIME_ASIDE ""
!insertmacro MOVE_RUNTIME_ASIDE "un."

; Move the previous version's python folder aside and delete it, so files a
; newer version no longer ships do not linger. Only a folder carrying
; OpenFlo's install marker is removed: a "python" folder some other program
; put in the chosen folder is left alone and the install stops.
Function RemoveOldRuntime
  ${IfNot} ${FileExists} "$INSTDIR\python\*.*"
    Return
  ${EndIf}
  ${IfNot} ${FileExists} "$INSTDIR\python\${MARKER}"
    MessageBox MB_OK|MB_ICONSTOP "The folder $INSTDIR already has a python folder that OpenFlo did not install. Choose another folder." /SD IDOK
    Abort "Install stopped: $INSTDIR\python was not installed by OpenFlo."
  ${EndIf}
  DetailPrint "Removing the previous OpenFlo version..."
  Call MoveRuntimeAside
  RMDir /r "$INSTDIR\python.old"
FunctionEnd

; Open files with extension EXT in OpenFlo, under the ProgID PROGID.
!macro REGISTER_TYPE EXT PROGID DESC
  WriteRegStr HKCU "${CLASSES}\${EXT}" "" "${PROGID}"
  WriteRegStr HKCU "${CLASSES}\${PROGID}" "" "${DESC}"
  WriteRegStr HKCU "${CLASSES}\${PROGID}\DefaultIcon" "" "$INSTDIR\OpenFlo.ico"
  WriteRegStr HKCU "${CLASSES}\${PROGID}\shell\open\command" "" '"$INSTDIR\python\pythonw.exe" -P -m ${GUI_MODULE} "%1"'
!macroend

; Undo REGISTER_TYPE: the ProgID always, the extension only while it still
; names that ProgID (another program may have claimed it since).
!macro UNREGISTER_TYPE EXT PROGID
  DeleteRegKey HKCU "${CLASSES}\${PROGID}"
  ReadRegStr $0 HKCU "${CLASSES}\${EXT}" ""
  ${If} $0 == "${PROGID}"
    DeleteRegValue HKCU "${CLASSES}\${EXT}" ""
    DeleteRegKey /ifempty HKCU "${CLASSES}\${EXT}"
  ${EndIf}
!macroend

Section "OpenFlo (required)" SecMain
  SectionIn RO
  Call RemoveOldRuntime
  SetOutPath "$INSTDIR"
  File "${PAYLOAD}\OpenFlo.ico"
  File /r "${PAYLOAD}\python"
  WriteUninstaller "$INSTDIR\${UNINSTALLER}"

  ; Compile the Python files now, so the first start is not slowed by it.
  ; The test folders are skipped; a failure here only costs start-up time.
  DetailPrint "Preparing OpenFlo for a quick first start (this can take a minute)..."
  nsExec::Exec '"$INSTDIR\python\python.exe" -m compileall -q -j 0 -x "[\\/](tests?|idlelib|turtledemo)[\\/]" "$INSTDIR\python\Lib"'
  Pop $0

  ; A shortcut's working folder is $OUTDIR: the user's home folder.
  SetOutPath "$PROFILE"
  CreateShortcut "$SMPROGRAMS\OpenFlo.lnk" "$INSTDIR\python\pythonw.exe" "-P -m ${GUI_MODULE}" "$INSTDIR\OpenFlo.ico" 0 SW_SHOWNORMAL "" "OpenFlo flow cytometry analysis"
  ; The Desktop section below puts it back when it is ticked.
  Delete "$DESKTOP\OpenFlo.lnk"

  WriteRegStr HKCU "${APP_KEY}" "InstallDir" "$INSTDIR"
  WriteRegDWORD HKCU "${APP_KEY}" "${UPDATES_VALUE}" $CheckUpdates
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayName" "${APP_NAME}"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayVersion" "${APP_VERSION}"
  WriteRegStr HKCU "${UNINST_KEY}" "Publisher" "@@PUBLISHER@@"
  WriteRegStr HKCU "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\OpenFlo.ico"
  WriteRegStr HKCU "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINST_KEY}" "URLInfoAbout" "@@ABOUT_URL@@"
  WriteRegStr HKCU "${UNINST_KEY}" "UninstallString" '"$INSTDIR\${UNINSTALLER}"'
  WriteRegStr HKCU "${UNINST_KEY}" "QuietUninstallString" '"$INSTDIR\${UNINSTALLER}" /S'
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "NoRepair" 1
  WriteRegDWORD HKCU "${UNINST_KEY}" "EstimatedSize" @@ESTIMATED_SIZE_KB@@

  !insertmacro REGISTER_TYPE ".flowsession" "OpenFlo.Session" "OpenFlo session"
  !insertmacro REGISTER_TYPE ".flowproject" "OpenFlo.Project" "OpenFlo project archive"
  ; Tell Explorer the associations changed (SHCNE_ASSOCCHANGED).
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
SectionEnd

Section "Desktop shortcut" SecDesktop
  SetOutPath "$PROFILE"
  CreateShortcut "$DESKTOP\OpenFlo.lnk" "$INSTDIR\python\pythonw.exe" "-P -m ${GUI_MODULE}" "$INSTDIR\OpenFlo.ico" 0 SW_SHOWNORMAL "" "OpenFlo flow cytometry analysis"
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain} "OpenFlo and the Python it runs on, and opening .flowsession and .flowproject files in OpenFlo. Nothing else on this PC is changed."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} "Put an OpenFlo shortcut on the Desktop."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Section "Uninstall"
  ; First, while nothing is deleted yet: stop if OpenFlo is running.
  ${If} ${FileExists} "$INSTDIR\python\${MARKER}"
    Call un.MoveRuntimeAside
  ${EndIf}
  Delete "$SMPROGRAMS\OpenFlo.lnk"
  Delete "$DESKTOP\OpenFlo.lnk"
  RMDir /r "$INSTDIR\python.old"
  Delete "$INSTDIR\OpenFlo.ico"
  Delete "$INSTDIR\${UNINSTALLER}"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "${UNINST_KEY}"
  DeleteRegKey HKCU "${APP_KEY}"
  !insertmacro UNREGISTER_TYPE ".flowsession" "OpenFlo.Session"
  !insertmacro UNREGISTER_TYPE ".flowproject" "OpenFlo.Project"
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
  DetailPrint "Your OpenFlo settings in $PROFILE\.openflo were kept."
SectionEnd
