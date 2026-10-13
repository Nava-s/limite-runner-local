# health-probe.ps1 - identify the HTTP service answering on the target port.
#
# Used by Avvia-Limite-GPU.cmd (Windows PowerShell 5.1, `powershell.exe`).
# It NEVER assumes that a failed probe means a free port: a timeout or an
# indeterminate transport error says nothing about who owns the port.
#
# Exit codes (the contract with the launcher):
#   0 = limite-runner answering and ready (2xx + service identity + status ok)
#   2 = limite-runner present but NOT ready (any HTTP status, identity match)
#   3 = another service answers (HTTP response without the runner identity)
#   1 = unreachable (connection refused / no listener): the port looks free
#   4 = timeout or indeterminate: the port may be busy or slow - do NOT start
#
# The body is parsed on EVERY HTTP status (including 503 and 404): an HTTP
# error response is a response, and only the identity decides who is there.

param(
    [Parameter(Mandatory = $true)][string]$Url,
    [int]$TimeoutSec = 3
)

$ErrorActionPreference = 'Stop'

function Read-ErrorBody($err) {
    # On HTTP error statuses Windows PowerShell keeps the body either in
    # ErrorDetails or in the response stream: try both.
    try {
        if ($err.ErrorDetails -ne $null -and $err.ErrorDetails.Message) {
            return [string]$err.ErrorDetails.Message
        }
    } catch {}
    try {
        $stream = $err.Exception.Response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($stream)
        $text = $reader.ReadToEnd()
        $reader.Close()
        return $text
    } catch {}
    return ''
}

function Get-ServiceName([string]$body) {
    try {
        $obj = $body | ConvertFrom-Json
        return @([string]$obj.service, [string]$obj.status)
    } catch {
        return @('', '')
    }
}

$status = 0
$body = ''
$haveResponse = $false

try {
    $resp = Invoke-WebRequest -Uri $Url -TimeoutSec $TimeoutSec -UseBasicParsing
    $status = [int]$resp.StatusCode
    $body = [string]$resp.Content
    $haveResponse = $true
} catch {
    $httpResp = $_.Exception.Response
    if ($httpResp -ne $null) {
        # An HTTP error status (404, 503, ...) is still a response: read it.
        $status = [int]$httpResp.StatusCode
        $body = Read-ErrorBody $_
        $haveResponse = $true
    } else {
        # No HTTP response at all: classify the transport failure from the
        # exception type/status codes -- never from localized message text.
        $isUncertain = $false
        $isRefused = $false
        # ConnectFailure is only a wrapper: require the specific socket
        # error, possibly nested below other exceptions. DNS failure does
        # not establish whether a service is listening on the target port.
        $failure = $_.Exception
        while ($failure -ne $null) {
            if ($failure -is [System.Net.WebException]) {
                if ($failure.Status -in @(
                    [System.Net.WebExceptionStatus]::Timeout,
                    [System.Net.WebExceptionStatus]::NameResolutionFailure,
                    [System.Net.WebExceptionStatus]::ProxyNameResolutionFailure
                )) { $isUncertain = $true }
            }
            if ($failure -is [System.Net.Sockets.SocketException]) {
                if ($failure.SocketErrorCode -eq [System.Net.Sockets.SocketError]::ConnectionRefused) {
                    $isRefused = $true
                } else {
                    $isUncertain = $true
                }
            }
            $failure = $failure.InnerException
        }
        if ($isRefused -and -not $isUncertain) { exit 1 }
        # DNS, timeout, reset, protocol errors: the port is NOT proven free.
        exit 4
    }
}

if (-not $haveResponse) { exit 4 }

$pair = Get-ServiceName $body
$service = $pair[0]
$bodyStatus = $pair[1]

if ($service -ne 'limite-runner') { exit 3 }
if ($status -ge 200 -and $status -lt 300 -and $bodyStatus -eq 'ok') { exit 0 }
exit 2
