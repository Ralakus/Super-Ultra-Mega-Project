use axum::{routing::get, Json, Router};
use clap::Parser;
use tower_http::services::{ServeDir, ServeFile};

use std::net::{IpAddr, Ipv6Addr, SocketAddr};

/// Command line arguments for server.
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Sets the port of the embedded webserver.
    #[clap()]
    port: u16,
}

/// Entry point for the server.
///
/// Parses arguments, sets up server-wide data,
/// sets up routes, and runs the axum http server.
///
/// ### Panics
/// Will panic if it fails to bind the interal
/// web server.
#[tokio::main]
async fn main() {
    // Command line parsing.
    let args = Args::parse();

    // HTTP server routes.
    let routes = Router::new()
        // Health endpoint for service monitors.
        .route("/health", get(|| async { Json("Ok") }))
        // Serve main page HTML and Javascript.
        .nest_service("/", ServeFile::new("index.html"))
        .nest_service("/index.js", ServeFile::new("index.js"))
        // Serve static files.
        .nest_service("/static", ServeDir::new("static"));

    // Bind socket and serve application.
    let socket = SocketAddr::new(IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1)), args.port);
    let listener = tokio::net::TcpListener::bind(socket).await.unwrap();
    axum::serve(listener, routes).await.unwrap();
}
