
#include "vp_split_node.h"
#include <iostream>

namespace vp_nodes {
        
    vp_split_node::vp_split_node(std::string node_name, 
                                bool split_with_channel_index,
                                bool split_with_deep_copy): 
                                vp_node(node_name), 
                                split_with_channel_index(split_with_channel_index),
                                split_with_deep_copy(split_with_deep_copy) {
        this->initialized();
    }
    
    vp_split_node::~vp_split_node() {
        deinitialized();
    }

    // override vp_meta_publisher::push_meta
    // total 2*2 mode to push meta
    void vp_split_node::push_meta(std::shared_ptr<vp_objects::vp_meta> meta) {
        if (this->split_with_channel_index) {
            std::shared_ptr<vp_meta_subscriber> next;
            auto channel_index = meta->channel_index;
            {
                std::lock_guard<std::mutex> guard(this->subscribers_lock);
                if (channel_index >= 0 && this->subscribers.size() > static_cast<std::size_t>(channel_index)) {
                    next = this->subscribers.at(channel_index);
                }
            }

            // assume array index stands for channel index
            // make sure the next nodes attached to current node in the order of channel index.
            if (next != nullptr) {
                if (this->split_with_deep_copy) {
                    meta = meta->clone();  // new pointer to new memory allocation in heap, old pointer is ignored.
                }
                next->meta_flow(meta);
            }
        }
        else {
            std::vector<std::shared_ptr<vp_meta_subscriber>> subscribers_snapshot;
            {
                std::lock_guard<std::mutex> guard(this->subscribers_lock);
                subscribers_snapshot = this->subscribers;
            }

            // see vp_meta_publisher::push_meta
            for (auto& subscriber: subscribers_snapshot) {
                if (this->split_with_deep_copy) {
                    meta = meta->clone(); // each next node has a new pointer to new memory allocation in heap
                }
                subscriber->meta_flow(meta);
            }
        }
    }
}
