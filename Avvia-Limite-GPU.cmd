@echo off
setlocal EnableExtensions
title Limite Violetto - chat locale

rem ==========================================================================
rem  Avvia-Limite-GPU.cmd - launcher portabile (Windows + WSL2 + CUDA)
rem
rem  Configurazione (precedenza):
rem    1. variabile d'ambiente gia' impostata
rem    2. file "limite-launcher.env" in questa cartella (vedi limite-launcher.env.example)
rem    3. valore predefinito elencato sotto
rem
rem  Variabili: LIMITE_WSL_DISTRO (default Ubuntu)
rem             LIMITE_HOST (127.0.0.1)   LIMITE_PORT (8080)
rem             LIMITE_GPU (1=GPU, 0=CPU) LIMITE_MAX_TOKENS (2048)
rem             LIMITE_QUANT (Q8_0)       LIMITE_CTX (16384)
rem             LIMITE_DATA_DIR, LIMITE_LLAMA_SERVER, LIMITE_MODEL
rem             LIMITE_WSL_REPO, LIMITE_BIN, LIMITE_LD_LIBRARY_PATH
rem             (ultimi 6: vuoti = predefinito Linux, vedi scripts/limite-serve.sh)
rem
rem  Uso:  Avvia-Limite-GPU.cmd              avvia server + chat
rem        Avvia-Limite-GPU.cmd --check      solo controlli requisiti
rem        Avvia-Limite-GPU.cmd --probe      solo la sonda della porta (exit code)
rem
rem  La porta viene riutilizzata solo se /health identifica un limite-runner
rem  (campo service="limite-runner"): un altro servizio sulla stessa porta
rem  (es. la vecchia chat llama-server nativa) produce un errore esplicito.
rem  Codici della sonda (scripts/health-probe.ps1, vedi anche --probe):
rem    0 = limite-runner pronto   2 = limite-runner non pronto
rem    3 = altro servizio          1 = non raggiungibile (porta libera)
rem    4 = timeout/incerto: la porta NON e' dimostrata libera, niente avvio.
rem ==========================================================================

set "SCRIPT_DIR=%~dp0"

rem -- 2. configurazione locale (non versionata) --
if exist "%SCRIPT_DIR%limite-launcher.env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%SCRIPT_DIR%limite-launcher.env") do (
    if not "%%~A"=="" if not defined %%~A set "%%~A=%%~B"
  )
)

rem -- 3. valori predefiniti --
if not defined LIMITE_WSL_DISTRO set "LIMITE_WSL_DISTRO=Ubuntu"
if not defined LIMITE_HOST set "LIMITE_HOST=127.0.0.1"
if not defined LIMITE_PORT set "LIMITE_PORT=8080"
if not defined LIMITE_GPU set "LIMITE_GPU=1"
if not defined LIMITE_QUANT set "LIMITE_QUANT=Q8_0"
if not defined LIMITE_CTX set "LIMITE_CTX=16384"
if not defined LIMITE_MAX_TOKENS set "LIMITE_MAX_TOKENS=2048"
if not defined LIMITE_LD_LIBRARY_PATH set "LIMITE_LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64"

rem A previous llama-ui at the same origin may still be cached by the browser.
rem Use a fresh document URL on every launch, including reuse of a live runner.
set "CHAT_URL=http://%LIMITE_HOST%:%LIMITE_PORT%/?limite-runner=%RANDOM%-%RANDOM%"

set "WIN_SCRIPT=%SCRIPT_DIR%scripts\limite-serve.sh"

if /i "%~1"=="--check" goto check
if /i "%~1"=="--probe" goto probe_only
if /i "%~1"=="--open-when-ready" goto open_when_ready
goto launch

rem ------------------------------------------------------------------------
:preflight
where wsl >nul 2>&1
if errorlevel 1 (
  echo ERRORE: WSL2 non trovato su questa macchina.
  echo Installa WSL2 con "wsl --install", riavvia e riprova.
  exit /b 1
)
rem Nota: wsl.exe di questo canale tratta le virgolette come parte del nome
rem distribuzione; il nome va passato senza virgolette (niente spazi nel nome).
echo %LIMITE_WSL_DISTRO%| findstr /c:" " >nul
if not errorlevel 1 (
  echo ERRORE: il nome della distribuzione WSL contiene spazi: "%LIMITE_WSL_DISTRO%"
  echo wsl.exe non accetta nomi con spazi da riga di comando in questo modo.
  echo Usa una distribuzione senza spazi, oppure lancia direttamente dentro WSL:
  echo   bash scripts/limite-serve.sh
  exit /b 1
)
wsl -d %LIMITE_WSL_DISTRO% --exec true >nul 2>&1
rem NOTA: wsl.exe puo' uscire con codice negativo (-1): "if errorlevel 1"
rem non lo intercetta, quindi si confronta esplicitamente con 0.
if %errorlevel% neq 0 (
  echo ERRORE: distribuzione WSL "%LIMITE_WSL_DISTRO%" non disponibile.
  echo Distribuzioni installate:
  wsl -l -v
  echo Imposta LIMITE_WSL_DISTRO in limite-launcher.env oppure nell'ambiente.
  exit /b 1
)
if not exist "%WIN_SCRIPT%" (
  echo ERRORE: non trovo "%WIN_SCRIPT%".
  echo Avvia il launcher dalla cartella completa della repository.
  exit /b 1
)
exit /b 0

rem ------------------------------------------------------------------------
:check
call :preflight
if errorlevel 1 exit /b 1
for /f "delims=" %%P in ('wsl -d %LIMITE_WSL_DISTRO% --exec wslpath -a "%WIN_SCRIPT%"') do set "WSL_SCRIPT=%%P"
echo Controllo requisiti in WSL "%LIMITE_WSL_DISTRO%"...
wsl -d %LIMITE_WSL_DISTRO% --exec env "LIMITE_DATA_DIR=%LIMITE_DATA_DIR%" "LIMITE_LLAMA_SERVER=%LIMITE_LLAMA_SERVER%" "LIMITE_MODEL=%LIMITE_MODEL%" "LIMITE_QUANT=%LIMITE_QUANT%" "LIMITE_CTX=%LIMITE_CTX%" "LIMITE_HOST=%LIMITE_HOST%" "LIMITE_PORT=%LIMITE_PORT%" "LIMITE_GPU=%LIMITE_GPU%" "LIMITE_MAX_TOKENS=%LIMITE_MAX_TOKENS%" "LIMITE_WSL_REPO=%LIMITE_WSL_REPO%" "LIMITE_BIN=%LIMITE_BIN%" "LIMITE_LD_LIBRARY_PATH=%LIMITE_LD_LIBRARY_PATH%" bash "%WSL_SCRIPT%" --check
exit /b %errorlevel%

rem ------------------------------------------------------------------------
:probe_health
rem Identita' del servizio sulla porta (mai riutilizzare un altro server).
rem La sonda (scripts/health-probe.ps1, Windows PowerShell) legge il corpo
rem anche su HTTP 4xx/5xx e distingue "risposta di errore" da "nessuna
rem connessione" e da "timeout":
rem   0 = limite-runner ok
rem   2 = limite-runner presente ma non pronto
rem   3 = risponde un ALTRO servizio (es. la vecchia chat llama-server nativa)
rem   1 = non raggiungibile (nessun ascoltatore: la porta risulta libera)
rem   4 = timeout/incerto: la porta NON e' dimostrata libera, niente avvio
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\health-probe.ps1" -Url "http://%LIMITE_HOST%:%LIMITE_PORT%/health"
exit /b %errorlevel%

:probe_only
call :probe_health
exit /b %errorlevel%

rem ------------------------------------------------------------------------
:open_when_ready
set /a _tries=0
:owl_loop
call :probe_health
set "PE=%errorlevel%"
if "%PE%"=="0" goto owl_open
if "%PE%"=="2" goto owl_open
if "%PE%"=="3" goto owl_done
rem 1 (non ancora in ascolto) e 4 (timeout/incerto): continua ad attendere
set /a _tries+=1
if %_tries% geq 36 goto owl_done
timeout /t 5 /nobreak >nul
goto owl_loop
:owl_open
start "" "%CHAT_URL%"
:owl_done
exit /b 0

rem ------------------------------------------------------------------------
:launch
echo.
echo Chat:   %CHAT_URL%
echo API:    http://%LIMITE_HOST%:%LIMITE_PORT%/v1
echo Lascia questa finestra aperta mentre usi il modello.
echo Premi Ctrl+C per fermare il server.
echo.
call :preflight
if errorlevel 1 (
  pause
  exit /b 1
)

call :probe_health
set "PE=%errorlevel%"
if "%PE%"=="0" goto reuse_ok
if "%PE%"=="2" goto reuse_notready
if "%PE%"=="3" goto port_busy
if "%PE%"=="4" goto probe_uncertain
rem 1 = porta non raggiungibile (nessun ascoltatore): posso avviare

for /f "delims=" %%P in ('wsl -d %LIMITE_WSL_DISTRO% --exec wslpath -a "%WIN_SCRIPT%"') do set "WSL_SCRIPT=%%P"

rem apre il browser appena /health del runner risponde (finestra ridotta a icona)
start "" /min "%~f0" --open-when-ready

wsl -d %LIMITE_WSL_DISTRO% --exec env "LIMITE_DATA_DIR=%LIMITE_DATA_DIR%" "LIMITE_LLAMA_SERVER=%LIMITE_LLAMA_SERVER%" "LIMITE_MODEL=%LIMITE_MODEL%" "LIMITE_QUANT=%LIMITE_QUANT%" "LIMITE_CTX=%LIMITE_CTX%" "LIMITE_HOST=%LIMITE_HOST%" "LIMITE_PORT=%LIMITE_PORT%" "LIMITE_GPU=%LIMITE_GPU%" "LIMITE_MAX_TOKENS=%LIMITE_MAX_TOKENS%" "LIMITE_WSL_REPO=%LIMITE_WSL_REPO%" "LIMITE_BIN=%LIMITE_BIN%" "LIMITE_LD_LIBRARY_PATH=%LIMITE_LD_LIBRARY_PATH%" bash "%WSL_SCRIPT%"

echo.
echo Server arrestato.
pause
exit /b 0

:reuse_ok
echo Trovato un limite-runner gia' attivo su %LIMITE_HOST%:%LIMITE_PORT%: riutilizzo quello.
start "" "%CHAT_URL%"
exit /b 0

:reuse_notready
echo Trovato un limite-runner su %LIMITE_HOST%:%LIMITE_PORT% ma non e' ancora pronto.
echo Apro comunque la chat: sara' disponibile a breve.
start "" "%CHAT_URL%"
exit /b 0

:port_busy
echo ERRORE: la porta %LIMITE_PORT% risponde ma NON e' limite-runner.
echo Su quella porta c'e' un altro servizio (es. la vecchia chat llama-server
echo nativa, che non applica il preprocessing single-turn): non la riutilizzo
echo e non avvio un secondo server sulla stessa porta.
echo Cambia porta con LIMITE_PORT (in limite-launcher.env o nell'ambiente)
echo oppure ferma quel servizio e riprova.
pause
exit /b 1

:probe_uncertain
echo ERRORE: http://%LIMITE_HOST%:%LIMITE_PORT%/health non ha dato una risposta
echo certa (timeout o errore di trasporto).
echo Un timeout NON dimostra che la porta sia libera: non avvio automaticamente.
echo Verifica che non ci sia un server occupato o molto lento su quella porta,
echo riprova fra poco, oppure cambia porta con LIMITE_PORT.
pause
exit /b 1
