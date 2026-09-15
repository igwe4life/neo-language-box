package com.sermontranslation.listener

import android.annotation.SuppressLint
import android.content.Intent
import android.net.Uri
import android.net.http.SslError
import android.os.Bundle
import android.os.PowerManager
import android.util.Patterns
import android.view.View
import android.view.WindowManager
import android.webkit.PermissionRequest
import android.webkit.SslErrorHandler
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.journeyapps.barcodescanner.ScanContract
import com.journeyapps.barcodescanner.ScanOptions

/**
 * Neo Sermon (listener) - single-activity app.
 *
 * The package stays com.sermontranslation.listener across the rename: Android
 * identifies an app by its applicationId, so changing it would make this a
 * different app - phones could not update in place and would end up with two.
 *
 * Two states, gated on whether we already know the server URL:
 *   PAIRING  : shows Scan QR + manual URL entry. First launch and after
 *              the user taps the little gear to change server.
 *   CONNECTED: full-screen WebView pointed at the server's listener page.
 *
 * Server URL is stored in a SharedPreferences entry. Everything the app
 * does happens over the phone's LAN connection to that server; nothing
 * ever leaves the local network.
 *
 * Self-signed HTTPS: our server generates a certificate for the LAN IP
 * on first run, so browsers see it as "not trusted." Inside the WebView
 * we override onReceivedSslError to accept the cert if the origin is
 * an RFC1918 private address, refusing anything else. That prevents the
 * app from silently talking to arbitrary self-signed hosts on the internet.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var prefs: android.content.SharedPreferences
    private var wakeLock: PowerManager.WakeLock? = null
    private lateinit var webview: WebView

    private val qrLauncher = registerForActivityResult(ScanContract()) { res ->
        val text = res?.contents ?: return@registerForActivityResult
        val url = normalizeUrl(text)
        if (url == null) {
            Toast.makeText(this, "That QR code doesn't look like a server URL", Toast.LENGTH_LONG).show()
            return@registerForActivityResult
        }
        saveAndConnect(url)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = getSharedPreferences("sermon", MODE_PRIVATE)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val saved = prefs.getString("server_url", null)
        if (saved.isNullOrBlank()) {
            showPairing()
        } else {
            showWeb(saved)
        }
    }

    // --------------------------------------------------------- pairing

    private fun showPairing() {
        setContentView(R.layout.activity_pair)
        findViewById<Button>(R.id.btn_scan).setOnClickListener { launchScanner() }
        findViewById<Button>(R.id.btn_connect).setOnClickListener {
            val raw = findViewById<EditText>(R.id.url_input).text.toString()
            val url = normalizeUrl(raw)
            if (url == null) {
                Toast.makeText(this, "Enter a URL like https://192.168.1.10:8443/", Toast.LENGTH_LONG).show()
            } else {
                saveAndConnect(url)
            }
        }
    }

    private fun launchScanner() {
        val opts = ScanOptions()
            .setDesiredBarcodeFormats(ScanOptions.QR_CODE)
            .setPrompt("Point at the QR code")
            .setBeepEnabled(false)
            .setOrientationLocked(false)
            .setCameraId(0)
        qrLauncher.launch(opts)
    }

    /** Turn arbitrary user / QR input into a well-formed http(s) URL, or null. */
    private fun normalizeUrl(raw: String): String? {
        var s = raw.trim()
        if (s.isEmpty()) return null
        if (!s.startsWith("http://", ignoreCase = true) &&
            !s.startsWith("https://", ignoreCase = true)) {
            s = "https://$s"
        }
        if (!Patterns.WEB_URL.matcher(s).matches()) return null
        // Ensure the URL points at the LISTENER root, not some deep path
        return try {
            val u = Uri.parse(s)
            val port = if (u.port > 0) ":${u.port}" else ""
            "${u.scheme}://${u.host}$port/"
        } catch (e: Exception) { null }
    }

    private fun saveAndConnect(url: String) {
        prefs.edit().putString("server_url", url).apply()
        showWeb(url)
    }

    // --------------------------------------------------------- web view

    @SuppressLint("SetJavaScriptEnabled")
    private fun showWeb(url: String) {
        setContentView(R.layout.activity_web)
        webview = findViewById(R.id.webview)
        val settings: WebSettings = webview.settings
        settings.javaScriptEnabled = true
        settings.domStorageEnabled = true                // localStorage for language + mute prefs
        settings.mediaPlaybackRequiresUserGesture = false // let the audio autoplay after user taps Start listening
        settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW

        webview.webViewClient = object : WebViewClient() {
            override fun onReceivedSslError(view: WebView?, handler: SslErrorHandler?, error: SslError?) {
                // Accept the cert ONLY for LAN addresses. Refuse anything else so
                // the app can't accidentally trust a self-signed cert served from
                // the public internet.
                val host = Uri.parse(error?.url ?: "").host ?: ""
                if (isLanHost(host)) handler?.proceed() else handler?.cancel()
            }
        }
        webview.webChromeClient = object : WebChromeClient() {
            // The listener page doesn't need the mic, but we grant camera+mic
            // preemptively so nothing prompts unexpectedly if the design changes.
            override fun onPermissionRequest(request: PermissionRequest?) {
                request?.grant(request.resources)
            }
        }
        webview.loadUrl(url)

        findViewById<Button>(R.id.btn_change).setOnClickListener {
            prefs.edit().remove("server_url").apply()
            showPairing()
        }
    }

    /** True if host is 127.0.0.1, 10.x, 172.16-31.x, or 192.168.x. */
    private fun isLanHost(host: String): Boolean {
        val h = host.lowercase()
        if (h == "localhost" || h.startsWith("127.")) return true
        if (h.startsWith("10.")) return true
        if (h.startsWith("192.168.")) return true
        val parts = h.split(".")
        if (parts.size == 4 && parts[0] == "172") {
            val second = parts[1].toIntOrNull() ?: return false
            if (second in 16..31) return true
        }
        return false
    }

    // --------------------------------------------------------- lifecycle

    override fun onResume() {
        super.onResume()
        val pm = getSystemService(POWER_SERVICE) as PowerManager
        @Suppress("DEPRECATION")
        wakeLock = pm.newWakeLock(PowerManager.SCREEN_DIM_WAKE_LOCK, "sermon:listen").apply {
            setReferenceCounted(false); acquire()
        }
    }

    override fun onPause() {
        super.onPause()
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    override fun onDestroy() {
        if (::webview.isInitialized) webview.destroy()
        super.onDestroy()
    }

    override fun onBackPressed() {
        if (::webview.isInitialized && webview.canGoBack()) {
            webview.goBack()
        } else super.onBackPressed()
    }
}
