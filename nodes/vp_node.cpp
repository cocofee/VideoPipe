
#include <exception>

#include "vp_node.h"

namespace vp_nodes {
    
    vp_node::vp_node(std::string node_name): node_name(node_name) {
    }
    
    vp_node::~vp_node() {

    }

    void vp_node::handle_run() {
        // cache for batch handling if need
        std::vector<std::shared_ptr<vp_objects::vp_frame_meta>> frame_meta_batch_cache;
        auto flush_frame_meta_batch = [this, &frame_meta_batch_cache]() {
            if (frame_meta_batch_cache.empty()) {
                return;
            }

            this->handle_frame_meta(frame_meta_batch_cache);
            if (node_type() != vp_node_type::DES) {
                for (auto& meta: frame_meta_batch_cache) {
                    pendding_meta(meta);
                }
            }
            frame_meta_batch_cache.clear();
        };

        while (true) {
            // wait for producer, make sure in_queue is not empty.
            this->in_queue_semaphore.wait();

            std::shared_ptr<vp_objects::vp_meta> in_meta;
            int in_queue_size_before = 0;
            int in_queue_size_after = 0;
            {
                std::lock_guard<std::mutex> guard(this->in_queue_lock);
                if (this->in_queue.empty()) {
                    continue;
                }
                in_queue_size_before = static_cast<int>(this->in_queue.size());
                in_meta = this->in_queue.front();
                this->in_queue.pop();
                in_queue_size_after = static_cast<int>(this->in_queue.size());
            }

            VP_DEBUG(vp_utils::string_format("[%s] before handling meta, in_queue.size()==>%d", node_name.c_str(), in_queue_size_before));

            // dead flag
            if (in_meta == nullptr) {
                flush_frame_meta_batch();
                pendding_meta(nullptr);
                break;
            }

            // handling hooker activated if need
            try {
                invoke_meta_handling_hooker(node_name, in_queue_size_before, in_meta);
            }
            catch (const std::exception& error) {
                VP_ERROR(vp_utils::string_format("[%s] meta handling hook failed: %s", node_name.c_str(), error.what()));
            }
            catch (...) {
                VP_ERROR(vp_utils::string_format("[%s] meta handling hook failed with unknown error", node_name.c_str()));
            }

            std::shared_ptr<vp_objects::vp_meta> out_meta;

            // call handlers
            if (in_meta->meta_type == vp_objects::vp_meta_type::CONTROL) {
                flush_frame_meta_batch();
                auto meta_2_handle = std::dynamic_pointer_cast<vp_objects::vp_control_meta>(in_meta);
                out_meta = this->handle_control_meta(meta_2_handle);
            }
            else if (in_meta->meta_type == vp_objects::vp_meta_type::FRAME) {    
                auto meta_2_handle = std::dynamic_pointer_cast<vp_objects::vp_frame_meta>(in_meta);
                // one by one
                if (frame_meta_handle_batch == 1) {                    
                    out_meta = this->handle_frame_meta(meta_2_handle);
                } 
                else {
                    // batch by batch
                    frame_meta_batch_cache.push_back(meta_2_handle);
                    if (frame_meta_batch_cache.size() >= frame_meta_handle_batch) {
                        flush_frame_meta_batch();
                    }
                    else {
                        // cache not complete, do nothing
                        VP_DEBUG(vp_utils::string_format("[%s] handle meta with batch, frame_meta_batch_cache.size()==>%d", node_name.c_str(), static_cast<int>(frame_meta_batch_cache.size())));
                    }
                }
            }
            else {
                throw "invalid meta type!";
            }
            VP_DEBUG(vp_utils::string_format("[%s] after handling meta, in_queue.size()==>%d", node_name.c_str(), in_queue_size_after));

            // one by one mode
            // return nullptr means do not push it to next nodes(such as in des nodes).
            if (out_meta != nullptr && node_type() != vp_node_type::DES) {
                pendding_meta(out_meta);
            }
        }
    }

    void vp_node::dispatch_run() {
        while (true) {
            // wait for producer, make sure out_queue is not empty.
            this->out_queue_semaphore.wait();

            std::shared_ptr<vp_objects::vp_meta> out_meta;
            int out_queue_size_before = 0;
            int out_queue_size_after = 0;
            {
                std::lock_guard<std::mutex> guard(this->out_queue_lock);
                if (this->out_queue.empty()) {
                    continue;
                }
                out_queue_size_before = static_cast<int>(this->out_queue.size());
                out_meta = this->out_queue.front();
                this->out_queue.pop();
                out_queue_size_after = static_cast<int>(this->out_queue.size());
            }

            VP_DEBUG(vp_utils::string_format("[%s] before dispatching meta, out_queue.size()==>%d", node_name.c_str(), out_queue_size_before));
            // dead flag
            if (out_meta == nullptr) {
                break;
            }

            // leaving hooker activated if need
            try {
                invoke_meta_leaving_hooker(node_name, out_queue_size_before, out_meta);
            }
            catch (const std::exception& error) {
                VP_ERROR(vp_utils::string_format("[%s] meta leaving hook failed: %s", node_name.c_str(), error.what()));
            }
            catch (...) {
                VP_ERROR(vp_utils::string_format("[%s] meta leaving hook failed with unknown error", node_name.c_str()));
            }

            // do something..
            this->push_meta(out_meta);
            VP_DEBUG(vp_utils::string_format("[%s] after dispatching meta, out_queue.size()==>%d", node_name.c_str(), out_queue_size_after));
        }
    }

    std::shared_ptr<vp_objects::vp_meta> vp_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        return meta;
    }

    std::shared_ptr<vp_objects::vp_meta> vp_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return meta;
    }

    void vp_node::handle_frame_meta(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& meta_with_batch) {
        
    }

    void vp_node::meta_flow(std::shared_ptr<vp_objects::vp_meta> meta) {
        if (meta == nullptr) {
            return;
        }

        bool drain_events = false;
        {
            std::lock_guard<std::mutex> event_guard(this->in_queue_event_lock);

            int in_queue_size_after = 0;
            {
                std::lock_guard<std::mutex> queue_guard(this->in_queue_lock);
                VP_DEBUG(vp_utils::string_format("[%s] before meta flow, in_queue.size()==>%d", node_name.c_str(), static_cast<int>(in_queue.size())));
                this->in_queue.push(meta);
                in_queue_size_after = static_cast<int>(this->in_queue.size());
            }

            this->in_queue_events.push({meta, in_queue_size_after});
            if (!this->in_queue_event_draining) {
                this->in_queue_event_draining = true;
                drain_events = true;
            }
        }

        if (!drain_events) {
            return;
        }

        std::exception_ptr first_error;
        while (true) {
            vp_in_queue_event event;
            {
                std::lock_guard<std::mutex> guard(this->in_queue_event_lock);
                if (this->in_queue_events.empty()) {
                    this->in_queue_event_draining = false;
                    break;
                }
                event = this->in_queue_events.front();
                this->in_queue_events.pop();
            }

            try {
                invoke_meta_arriving_hooker(node_name, event.queue_size_after, event.meta);
            }
            catch (...) {
                if (first_error == nullptr) {
                    first_error = std::current_exception();
                }
            }

            this->in_queue_semaphore.signal();
            VP_DEBUG(vp_utils::string_format("[%s] after meta flow, in_queue.size()==>%d", node_name.c_str(), event.queue_size_after));
        }

        if (first_error != nullptr) {
            std::rethrow_exception(first_error);
        }
    }

    void vp_node::detach() {
        auto self = shared_from_this();
        for (auto& pre_node_ref: this->pre_nodes) {
            if (auto pre_node = pre_node_ref.lock()) {
                pre_node->remove_subscriber(self);
            }
        }
        this->pre_nodes.clear();
    }

    void vp_node::detach_from(std::vector<std::string> pre_node_names) {
        auto self = shared_from_this();
        for (auto i = this->pre_nodes.begin(); i != this->pre_nodes.end();) {
            auto pre_node = i->lock();
            if (pre_node == nullptr) {
                i = this->pre_nodes.erase(i);
            }
            else if (std::find(pre_node_names.begin(), pre_node_names.end(), pre_node->node_name) != pre_node_names.end()) {
                pre_node->remove_subscriber(self);
                i = this->pre_nodes.erase(i);
            }
            else {
                i++;
            }
        }
    }

    void vp_node::detach_recursively() {
        detach();
        auto nodes = next_nodes();
        for (auto& n: nodes) {
            n->detach_recursively();
        }
    }

    void vp_node::attach_to(std::vector<std::shared_ptr<vp_node>> pre_nodes) {
        // can not attach src node to any previous nodes
        if (this->node_type() == vp_node_type::SRC) {
            throw vp_excepts::vp_invalid_calling_error("SRC nodes must not have any previous nodes!");
        }
        // can not attach any nodes to des node
        for(auto i : pre_nodes) {
            if (i->node_type() == vp_node_type::DES) {
                throw vp_excepts::vp_invalid_calling_error("DES nodes must not have any next nodes!");
            }
            i->add_subscriber(shared_from_this());
            this->pre_nodes.push_back(i);
        }
    }

    void vp_node::initialized() {
        // start threads since all resources have been initialized
        this->handle_thread = std::thread(&vp_node::handle_run, this);
        this->dispatch_thread = std::thread(&vp_node::dispatch_run, this);
    }

    void vp_node::deinitialized() {
        // send dead flag
        alive.store(false);
        {
            std::lock_guard<std::mutex> guard(this->in_queue_lock);
            this->in_queue.push(nullptr);
        }
        this->in_queue_semaphore.signal();
        // wait for threads exits in vp_node
        if (handle_thread.joinable()) {
            handle_thread.join();
        }
        if (dispatch_thread.joinable()) {
            dispatch_thread.join();
        }
    }

    vp_node_type vp_node::node_type() {
        // return vp_node_type::MID by default
        // need override in child class
        return vp_node_type::MID;
    }

    std::vector<std::shared_ptr<vp_node>> vp_node::next_nodes() {
        std::vector<std::shared_ptr<vp_node>> next_nodes;
        std::lock_guard<std::mutex> guard(this->subscribers_lock);
        for(auto & i: this->subscribers) {
            next_nodes.push_back(std::dynamic_pointer_cast<vp_node>(i));
        }
        return next_nodes;
    }

    std::string vp_node::to_string() {
        // return node_name by default
        return node_name;
    }

    void vp_node::pendding_meta(std::shared_ptr<vp_objects::vp_meta> meta) {
        bool drain_events = false;
        {
            std::lock_guard<std::mutex> event_guard(this->out_queue_event_lock);

            int out_queue_size_before = 0;
            int out_queue_size_after = 0;
            {
                std::lock_guard<std::mutex> queue_guard(this->out_queue_lock);
                out_queue_size_before = static_cast<int>(this->out_queue.size());
                this->out_queue.push(meta);
                out_queue_size_after = static_cast<int>(this->out_queue.size());
            }

            this->out_queue_events.push({meta, out_queue_size_before, out_queue_size_after});
            if (!this->out_queue_event_draining) {
                this->out_queue_event_draining = true;
                drain_events = true;
            }
        }

        if (!drain_events) {
            return;
        }

        std::exception_ptr first_error;
        while (true) {
            vp_out_queue_event event;
            {
                std::lock_guard<std::mutex> guard(this->out_queue_event_lock);
                if (this->out_queue_events.empty()) {
                    this->out_queue_event_draining = false;
                    break;
                }
                event = this->out_queue_events.front();
                this->out_queue_events.pop();
            }

            if (event.meta != nullptr) {
                try {
                    VP_DEBUG(vp_utils::string_format("[%s] before handling meta, out_queue.size()==>%d", node_name.c_str(), event.queue_size_before));
                    // handled hooker activated if need
                    invoke_meta_handled_hooker(node_name, event.queue_size_after, event.meta);
                    VP_DEBUG(vp_utils::string_format("[%s] after handling meta, out_queue.size()==>%d", node_name.c_str(), event.queue_size_after));
                }
                catch (...) {
                    if (first_error == nullptr) {
                        first_error = std::current_exception();
                    }
                }
            }

            // notify consumer of out_queue
            this->out_queue_semaphore.signal();
        }

        if (first_error != nullptr) {
            std::rethrow_exception(first_error);
        }
    }
}
