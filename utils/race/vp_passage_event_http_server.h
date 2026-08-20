#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "vp_passage_event_ingestor.h"

namespace vp_utils {

    // LAN HTTP adapter for CycleRace passage events.
    class vp_passage_event_http_server {
    private:
        class implementation;

        std::string host;
        int port = 0;
        vp_passage_event_ingestor& ingestor;
        std::unique_ptr<implementation> server;
        std::thread server_thread;
        std::atomic<bool> started = false;

        void run();

    public:
        vp_passage_event_http_server(std::string host, int port, vp_passage_event_ingestor& ingestor);
        ~vp_passage_event_http_server();

        vp_passage_event_http_server(const vp_passage_event_http_server&) = delete;
        vp_passage_event_http_server& operator=(const vp_passage_event_http_server&) = delete;

        void start();
        void stop();
        bool is_running() const;
        const std::string& listen_host() const;
        int listen_port() const;
    };

}
