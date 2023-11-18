//
//  EventController.swift
//  stockInterface
//
//  Created by CHONK on 11/12/23.
//

import Cocoa

class EventController {
    static let shared = EventController()
    
    func handleEvent(event: NSEvent) {
        if event.type == .keyDown {
            switch event.keyCode {
            case 123: // left arrow
                PlaybackController.shared.addToPlaybackIndex(-1)
            case 124: // right arrow
                PlaybackController.shared.addToPlaybackIndex(1)
            case 125: // down arrow
                PlaybackController.shared.addToPlaybackIndex(-10)
            case 126: // up arrow
                PlaybackController.shared.addToPlaybackIndex(10)
            default: break
            }
        }
    }
}
