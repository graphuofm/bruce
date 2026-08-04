//! bruce-server library surface.
//!
//! The package ships two binaries and one library:
//!   - bin `bruce-server` (src/main.rs): the KvMemory HTTP surface.
//!     It is self-contained and does NOT use this library.
//!   - bin `bruce-flight-server` (src/bin/flight_server.rs): the
//!     Arrow Flight surface over the bruce-query Database.
//!   - this lib: the Flight service implementation ([`flight`]) and
//!     the npy key-matrix reader ([`npy`]), exported so integration
//!     tests can serve the service in-process.

pub mod flight;
pub mod npy;
