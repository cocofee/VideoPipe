#include "vp_passage_event_http_server.h"

#include <stdexcept>
#include <utility>

#include "../../third_party/cpp_httplib/httplib.h"
#include "../../third_party/nlohmann/json.hpp"

namespace vp_utils {

    class vp_passage_event_http_server::implementation {
    public:
        httplib::Server http;
    };

    vp_passage_event_http_server::vp_passage_event_http_server(
        std::string host,
        int port,
        vp_passage_event_ingestor& ingestor):
        host(std::move(host)),
        port(port),
        ingestor(ingestor),
        server(std::make_unique<implementation>()) {
        if (this->host.empty()) {
            throw std::invalid_argument("passage event HTTP server host is required");
        }
        if (this->port <= 0 || this->port > 65535) {
            throw std::invalid_argument("passage event HTTP server port is out of range");
        }

        httplib::Server::Handler handler = [this](const httplib::Request& request, httplib::Response& response) {
            try {
                const auto result = this->ingestor.ingest_json(request.body);
                nlohmann::json body = {
                    {"schema_version", vp_objects::VP_PASSAGE_EVENT_SCHEMA_VERSION},
                    {"message_type", "passage_ack"},
                    {"status", result == vp_passage_event_ingest_result::accepted ? "accepted" : "duplicate"},
                };
                response.status = result == vp_passage_event_ingest_result::accepted ? 201 : 200;
                response.set_content(body.dump(), "application/json");
            }
            catch (const vp_passage_event_conflict_error& error) {
                nlohmann::json body = {
                    {"schema_version", vp_objects::VP_PASSAGE_EVENT_SCHEMA_VERSION},
                    {"message_type", "passage_ack"},
                    {"status", "rejected"},
                    {"error", error.what()},
                };
                response.status = 409;
                response.set_content(body.dump(), "application/json");
            }
            catch (const nlohmann::json::exception& error) {
                nlohmann::json body = {
                    {"schema_version", vp_objects::VP_PASSAGE_EVENT_SCHEMA_VERSION},
                    {"message_type", "passage_ack"},
                    {"status", "rejected"},
                    {"error", error.what()},
                };
                response.status = 400;
                response.set_content(body.dump(), "application/json");
            }
            catch (const std::invalid_argument& error) {
                nlohmann::json body = {
                    {"schema_version", vp_objects::VP_PASSAGE_EVENT_SCHEMA_VERSION},
                    {"message_type", "passage_ack"},
                    {"status", "rejected"},
                    {"error", error.what()},
                };
                response.status = 400;
                response.set_content(body.dump(), "application/json");
            }
            catch (const vp_passage_event_delivery_error& error) {
                nlohmann::json body = {
                    {"schema_version", vp_objects::VP_PASSAGE_EVENT_SCHEMA_VERSION},
                    {"message_type", "passage_ack"},
                    {"status", "retry"},
                    {"error", error.what()},
                };
                response.status = 503;
                response.set_content(body.dump(), "application/json");
            }
            catch (const std::exception& error) {
                nlohmann::json body = {
                    {"schema_version", vp_objects::VP_PASSAGE_EVENT_SCHEMA_VERSION},
                    {"message_type", "passage_ack"},
                    {"status", "error"},
                    {"error", error.what()},
                };
                response.status = 500;
                response.set_content(body.dump(), "application/json");
            }
        };
        server->http.Post("/api/v1/passage-events", std::move(handler));
    }

    vp_passage_event_http_server::~vp_passage_event_http_server() {
        stop();
    }

    void vp_passage_event_http_server::start() {
        if (started.exchange(true)) {
            return;
        }

        if (!server->http.bind_to_port(host, port)) {
            started = false;
            throw std::runtime_error("failed to bind passage event HTTP server to " + host + ":" + std::to_string(port));
        }

        server_thread = std::thread(&vp_passage_event_http_server::run, this);
        server->http.wait_until_ready();
    }

    void vp_passage_event_http_server::run() {
        server->http.listen_after_bind();
        started = false;
    }

    void vp_passage_event_http_server::stop() {
        if (server) {
            server->http.stop();
        }
        if (server_thread.joinable()) {
            server_thread.join();
        }
        started = false;
    }

    bool vp_passage_event_http_server::is_running() const {
        return server && server->http.is_running();
    }

    const std::string& vp_passage_event_http_server::listen_host() const {
        return host;
    }

    int vp_passage_event_http_server::listen_port() const {
        return port;
    }

}
