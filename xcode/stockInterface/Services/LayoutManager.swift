//
//  LayoutManager.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 10/13/23.
//

import Cocoa

class LayoutManager {
    
    static let shared = LayoutManager()
    
    // MARK: Properties
    
    var celledMasterView: CelledMasterView!
    
    
    
    // MARK: Init
    
    init() {
        self.celledMasterView = CelledMasterView()
    }
    
    // MARK: Getters Setters
    
    // Used when initializing window
    func getCelledMasterView() -> NSView { return celledMasterView }
    
    // MARK: TBD
    
    // MARK: Private


}
