//
//  CoreDataStack.swift
//  stockInterface
//
//  Created by CHONK on 9/29/23.
//

import Foundation
import CoreData

class CoreDataStack {
    
    static let Container: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "StockData")
        container.loadPersistentStores(completionHandler: { (_, error) in
            if let error = error{
                fatalError("Failed to Load Persistent Store \(error)")
            }
        })
        return container
    }()
    
    static var Context: NSManagedObjectContext {
        return Container.viewContext
    }
    
    static var TempContext: NSManagedObjectContext = {
        let context = NSManagedObjectContext(concurrencyType: .mainQueueConcurrencyType)
        context.parent = CoreDataStack.Context
        return context
    }()
    
    // Save If Needed
    static func SaveToPersistentStore() {
        if CoreDataStack.Context.hasChanges {
            try? CoreDataStack.Context.save()
        }
    }
}
