//
//  LoadPlayblackCommand.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/12/23.
//

import Foundation

class LoadPlayblackCommand: Command {
    var name: String { "load" }

    func execute(with arguments: [String]) {
        PlaybackController.shared.loadPlaybackFromFile()
    }
}
