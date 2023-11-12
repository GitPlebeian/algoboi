//
//  Command.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/11/23.
//

import Foundation

protocol Command {
    var name: String { get }
    func execute(with arguments: [String])
}
